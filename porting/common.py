"""Shared helpers for the Metalhead porting scripts."""
import contextlib
import copy
import functools
import importlib
import inspect
import json
import math
import sys
import types

import numpy as np
import timm
import torch
from torch import nn


def reference_input(shape):
    """Deterministic input, reproduced exactly by `reference_input` in test/parity.jl."""
    b, c, h, w = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    k = (b * 131 + c * 71 + h * 37 + w * 17) % 97
    return torch.from_numpy((k.astype(np.float64) / 24 - 2).astype(np.float32))


def build(spec, **extra):
    """Build the model described by a spec: `timm.create_model` or a named builder.

    With `"shrink": k` in the spec, every channel count is divided by k (see `shrunk_channels`).
    A spec with `"library": "transformers"` builds a transformers model (see `build_transformers`),
    and one with `"library": "diffusers"` a diffusers model (see `build_diffusers`).
    """
    if spec.get("library") == "transformers":
        return build_transformers(spec)
    if spec.get("library") == "diffusers":
        return build_diffusers(spec)
    kwargs = copy.deepcopy(dict(spec.get("kwargs", {}), **extra))  # some builders mutate them
    shrink = spec.get("shrink")
    keep = {spec["input_shape"][1], kwargs.get("num_classes", 1000)} if shrink else set()
    with shrunk_channels(shrink, keep) if shrink else contextlib.nullcontext():
        if spec.get("builder"):
            module, fn = spec["builder"].split(":")
            model = getattr(importlib.import_module(module), fn)(**kwargs)
        else:
            model = timm.create_model(spec["architecture"], pretrained=False, **kwargs)
    if shrink:
        unknown = {type(m).__name__ for m in model.modules() if not isinstance(m, SHRUNK)
                   and any(True for _ in m.parameters(recurse=False))}
        assert not unknown, f"shrink does not handle the parameters of {sorted(unknown)}"
    return model


def build_transformers(spec):
    """The transformers class `spec["architecture"]`, built from a config of `spec["model_type"]`
    with the fields `spec["config"]`, with PyTorch's scaled_dot_product_attention and the eager
    loop over the experts of a mixture of experts."""
    import transformers
    config = transformers.AutoConfig.for_model(spec["model_type"], **spec.get("config", {}))
    return getattr(transformers, spec["architecture"])._from_config(
        config, attn_implementation="sdpa", experts_implementation="eager")


def build_diffusers(spec):
    """The diffusers model class `spec["architecture"]`, built from the arguments `spec["config"]`."""
    import diffusers
    return getattr(diffusers, spec["architecture"]).from_config(spec.get("config", {}))


SHRUNK = (nn.Conv2d, nn.modules.batchnorm._NormBase, nn.Linear)


@contextlib.contextmanager
def shrunk_channels(k, keep=()):
    """Build models with every channel count divided by `k`, for fixtures of fixed-width models.

    Patches the constructors of `nn.Conv2d` (in and out channels), `nn.BatchNorm*d` (features)
    and `nn.Linear` (in and out features), subclasses included. A count in `keep` stays as is
    (the image channels and `num_classes`), every other count must be divisible by `k`.
    A depthwise conv (groups == in_channels) keeps groups == in_channels; other `groups` stay.
    """
    def shrink(n):
        if n in keep:
            return n
        assert n % k == 0, f"channel count {n} is not divisible by shrink = {k}"
        return n // k

    def patch(cls, fn):
        init = cls.__init__
        sig = inspect.signature(init)

        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            fn(bound.arguments)
            return init(*bound.args, **bound.kwargs)
        cls.__init__ = wrapper
        return lambda: setattr(cls, "__init__", init)

    def conv(a):
        depthwise = a.get("groups", 1) > 1 and a.get("groups") == a["in_channels"]
        a["in_channels"], a["out_channels"] = shrink(a["in_channels"]), shrink(a["out_channels"])
        if depthwise:
            a["groups"] = a["in_channels"]

    def norm(a):
        a["num_features"] = shrink(a["num_features"])

    def linear(a):
        a["in_features"], a["out_features"] = shrink(a["in_features"]), shrink(a["out_features"])

    restores = [patch(nn.Conv2d, conv), patch(nn.modules.batchnorm._NormBase, norm),
                patch(nn.Linear, linear)]
    try:
        yield
    finally:
        for restore in reversed(restores):
            restore()


@contextlib.contextmanager
def float64_pos_embed():
    """Resample ViT position embeddings in the precision of the model.

    timm's `resample_abs_pos_embed` (dynamic image sizes) interpolates in float32 even for a
    float64 model, which would cap the precision of float64 fixtures. This runs its exact steps
    without the cast; the Julia port computes in the eltype of the embedding too.
    """
    import torch.nn.functional as F
    from timm.models import vision_transformer as vit

    original = vit.resample_abs_pos_embed

    def resample(posemb, new_size, old_size=None, num_prefix_tokens=1,
                 interpolation="bicubic", antialias=True, verbose=False):
        if posemb.dtype != torch.float64:
            return original(posemb, new_size, old_size, num_prefix_tokens, interpolation,
                            antialias, verbose)
        num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        if num_new_tokens == posemb.shape[1] and (
                (old_size is not None and tuple(old_size) == tuple(new_size))
                or (old_size is None and new_size[0] == new_size[1])):
            return posemb
        if old_size is None:
            hw = int((posemb.shape[1] - num_prefix_tokens) ** 0.5)
            old_size = hw, hw
        prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
        dim = posemb.shape[-1]
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([prefix, posemb], dim=1) if num_prefix_tokens else posemb

    vit.resample_abs_pos_embed = resample
    try:
        yield
    finally:
        vit.resample_abs_pos_embed = original


NORMS = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)


@torch.no_grad()
def rerandomize(model, seed=0):
    """Give every parameter and BN statistic a random value.

    Default inits (zero_init_last, tiny layer-scale, unit BN stats) would hide bugs in whole branches.
    """
    gen = torch.Generator().manual_seed(seed)

    def normal(t, std, mean=0.0):
        t.copy_(torch.randn(t.shape, generator=gen, dtype=torch.float64) * std + mean)

    def uniform(t, lo, hi):
        t.copy_(torch.rand(t.shape, generator=gen, dtype=torch.float64) * (hi - lo) + lo)

    for module in model.modules():
        is_norm = isinstance(module, NORMS)
        for name, t in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
            if not t.is_floating_point():
                continue
            if is_norm:
                if name == "weight":
                    uniform(t, 0.7, 1.3)
                elif name == "bias":
                    normal(t, 0.1)
                elif name == "running_mean":
                    normal(t, 0.2)
                elif name == "running_var":
                    uniform(t, 0.5, 1.5)
            elif name in module._buffers:
                continue
            elif name == "bias":
                normal(t, 0.1)
            elif t.ndim >= 2 and name == "weight":
                normal(t, t[0].numel() ** -0.5)
            elif t.ndim == 3 and name.endswith("_proj"):  # stacked experts (num_experts, out, in)
                normal(t, t.shape[-1] ** -0.5)
            elif name == "A_log":  # decay rates of linear attention, slow enough to carry
                uniform(t, -6, 0)  # the recurrent state across chunks of 64 tokens
            elif t.ndim <= 1:
                uniform(t, 0.5, 1.5)
            else:
                normal(t, 0.5)


def exact_float64(model):
    """Compute the RMSNorms, rotary tables, routers and gated delta rules of a transformers
    model in its own dtype.

    transformers computes them in float32 even in a float64 model, which would limit the
    fixtures to float32 precision. In a float32 model the results are unchanged but for the
    rounding of the rotary frequencies, which are computed in float64 here, and of the weights
    that interpolate the position embeddings of a Qwen3.5 vision encoder (see
    `_exact_float64_vision`).
    """
    for m in model.modules():
        name = type(m).__name__
        if name.endswith("RMSNorm"):
            zero_centred = name.startswith(("Gemma", "Qwen3_5"))  # scale by 1 + weight
            forward = _zero_centred_rmsnorm_forward if zero_centred else _rmsnorm_forward
            m.forward = types.MethodType(forward, m)
        elif name.endswith("RMSNormGated"):
            m.forward = types.MethodType(_gated_rmsnorm_forward, m)
        elif name.endswith("GatedDeltaNet"):
            m.forward = types.MethodType(_gated_delta_net_forward, m)
        elif name.startswith("GraniteMoe") and name.endswith("TopKRouter"):
            m.forward = types.MethodType(_granitemoe_router_forward, m)
        elif name.endswith("VisionRotaryEmbedding"):  # Qwen3.5's 2D axial rotary embedding
            m.exact_inv_freq = _exact_axial_inv_freq(m)
            m.forward = types.MethodType(_axial_rotary_forward, m)
            _exact_float64_vision(sys.modules[type(m).__module__])
        elif name == "Sam3ViTRotaryEmbedding":  # SAM 3's, which rotates pairs of features
            m.exact_inv_freq = _exact_axial_inv_freq(m)
            m.forward = types.MethodType(_axial_rotary_forward, m)
            _exact_float64_sam3(sys.modules[type(m).__module__])
        elif name == "Sam3ViTLayer":
            m.position_ids = _exact_sam3_positions(m)
        elif name.endswith("RotaryEmbedding"):
            if isinstance(m.rope_type, dict):  # a table for each layer type (Gemma 3)
                m.exact_inv_freq = {t: _exact_inv_freq(m.config.rope_parameters[t],
                                                       getattr(m, f"{t}_inv_freq"))
                                    for t in m.rope_type}
                m.forward = types.MethodType(_layer_type_rotary_forward, m)
            else:
                m.exact_inv_freq = _exact_inv_freq(m.config.rope_parameters, m.inv_freq)
                mrope = hasattr(m, "mrope_section") and hasattr(m, "recomposition_frequencies")
                forward = _mrope_rotary_forward if mrope else _rotary_forward
                m.forward = types.MethodType(forward, m)


def _rmsnorm_forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (x * torch.rsqrt(variance + self.variance_epsilon))


def _granitemoe_router_forward(self, hidden_states):
    router_logits = torch.nn.functional.linear(hidden_states, self.weight)
    top_k_logits, top_k_index = router_logits.topk(self.top_k, dim=-1)
    return top_k_index, torch.softmax(top_k_logits, dim=-1), router_logits


def _zero_centred_rmsnorm_forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + self.eps)) * (1.0 + self.weight)


def _gated_rmsnorm_forward(self, hidden_states, gate):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))
    return hidden_states * torch.nn.functional.silu(gate)


def _gated_delta_net_forward(self, hidden_states, cache_params=None, attention_mask=None,
                             **kwargs):
    """`Qwen3_5GatedDeltaNet.forward` on a whole sequence (a prefill, which leaves the cache as
    it is), with the decays `g` and the chunked gated delta rule computed in the dtype of the
    model rather than in float32."""
    assert cache_params is None or not cache_params.has_previous_state(self.layer_idx, 0)
    modeling = sys.modules[type(self).__module__]
    hidden_states = modeling.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = modeling.causal_conv1d_fn(
        self.in_proj_qkv(hidden_states).transpose(1, 2), self.conv1d.weight.squeeze(1),
        self.conv1d.bias, activation=self.activation).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim],
                                    dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    a = self.in_proj_a(hidden_states)
    g = -self.A_log.exp() * torch.nn.functional.softplus(a + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    rule = modeling.torch_chunk_gated_delta_rule
    if query.dtype == torch.float64:
        rule = _float32_as_float64(rule)
    core_attn_out, _ = rule(query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim),
                              z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


class _Float32AsFloat64:
    """The `torch` module, but for `torch.float32`, which is `torch.float64`."""

    def __getattr__(self, name):
        return torch.float64 if name == "float32" else getattr(torch, name)


def _float32_as_float64(fn):
    """The function `fn` of a transformers module, undecorated, with `torch.float32` read as
    `torch.float64`: transformers' own code, computing in float64 where it casts to float32."""
    fn = inspect.unwrap(fn)
    new = types.FunctionType(fn.__code__, {**fn.__globals__, "torch": _Float32AsFloat64()},
                             fn.__name__, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    return new


def _exact_inv_freq(params, inv_freq):
    """The `inv_freq` of the rotary embedding with `rope_parameters` `params` in float64, which
    must match transformers' own float32 `inv_freq` to float32 precision."""
    dim = 2 * inv_freq.numel()
    freqs = 1.0 / (params["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    rope_type = params.get("rope_type", "default")
    if rope_type == "linear":
        freqs = freqs / params["factor"]
    elif rope_type == "llama3":  # `_compute_llama3_parameters`
        factor, low, high = params["factor"], params["low_freq_factor"], params["high_freq_factor"]
        old_context_len = params["original_max_position_embeddings"]
        wavelen = 2 * math.pi / freqs
        smooth = (old_context_len / wavelen - low) / (high - low)
        medium = (1 - smooth) * freqs / factor + smooth * freqs
        freqs = torch.where(wavelen < old_context_len / high, freqs,
                            torch.where(wavelen > old_context_len / low, freqs / factor, medium))
    else:
        assert rope_type == "default", f"rope_type {rope_type} is not supported"
    assert torch.allclose(freqs, inv_freq.double(), rtol=1e-6, atol=0), (freqs, inv_freq)
    return freqs


@torch.no_grad()
def _rotary_forward(self, x, position_ids):
    return _rotary_tables(x, position_ids, self.exact_inv_freq, self.attention_scaling)


@torch.no_grad()
def _mrope_rotary_forward(self, x, position_ids):
    """The multimodal rotary tables of Qwen3.5, for position ids of size (3, batch, seq_len):
    the frequencies of the temporal, height and width positions, recomposed by the model's
    own `recomposition_frequencies`."""
    freqs = position_ids.expand(3, -1, -1)[..., None].double() * self.exact_inv_freq
    cos = self.recomposition_frequencies(freqs.cos() * self.attention_scaling)
    sin = self.recomposition_frequencies(freqs.sin() * self.attention_scaling)
    return cos.to(x.dtype), sin.to(x.dtype)


@torch.no_grad()
def _layer_type_rotary_forward(self, x, position_ids, layer_type):
    return _rotary_tables(x, position_ids, self.exact_inv_freq[layer_type],
                          getattr(self, f"{layer_type}_attention_scaling"))


def _rotary_tables(x, position_ids, inv_freq, attention_scaling):
    freqs = position_ids[..., None].double() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return ((emb.cos() * attention_scaling).to(x.dtype),
            (emb.sin() * attention_scaling).to(x.dtype))


def _exact_axial_inv_freq(m):
    """The `inv_freq` of a `Qwen3_5VisionRotaryEmbedding` in float64."""
    config = m.config
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    spatial = dim // 2
    freqs = 1.0 / (config.rope_parameters["rope_theta"]
                   ** (torch.arange(0, spatial, 2, dtype=torch.float64) / spatial))
    assert torch.allclose(freqs, m.inv_freq.double(), rtol=1e-6, atol=0), (freqs, m.inv_freq)
    return freqs


@torch.no_grad()
def _axial_rotary_forward(self, x, position_ids):
    freqs = position_ids[..., None].double() * self.exact_inv_freq
    return (self.recomposition_frequencies(freqs.cos() * self.attention_scaling),
            self.recomposition_frequencies(freqs.sin() * self.attention_scaling))


def _exact_float64_vision(module):
    """Patch the vision encoder of `module` (`modeling_qwen3_5`) to rotate float64 queries and
    keys in float64, and `vision_utils` to compute the weights that interpolate the learned
    position embeddings in float64. transformers computes both in float32."""
    from transformers import vision_utils
    if getattr(module, "_metalhead_exact_float64", False):
        return
    original = module.apply_rotary_pos_emb_vision

    def apply_rotary_pos_emb_vision(q, k, cos, sin):
        if q.dtype != torch.float64:
            return original(q, k, cos, sin)
        cos, sin = cos.unsqueeze(-2).to(q.dtype), sin.unsqueeze(-2).to(q.dtype)
        return (q * cos + module.rotate_half(q) * sin, k * cos + module.rotate_half(k) * sin)

    module.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision
    vision_utils._interpolation_axis_taps_weights = _interpolation_axis_taps_weights_float64
    module._metalhead_exact_float64 = True


def _exact_float64_sam3(module):
    """Patch the ViT of `module` (`modeling_sam3`) to rotate float64 queries and keys in
    float64. transformers rotates them in float32."""
    if getattr(module, "_metalhead_exact_float64", False):
        return
    original = module.apply_rotary_pos_emb_2d

    def apply_rotary_pos_emb_2d(q, k, cos, sin):
        if q.dtype != torch.float64:
            return original(q, k, cos, sin)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        return (q * cos + module.rotate_pairwise(q) * sin,
                k * cos + module.rotate_pairwise(k) * sin)

    module.apply_rotary_pos_emb_2d = apply_rotary_pos_emb_2d
    module._metalhead_exact_float64 = True


def _exact_sam3_positions(layer):
    """The rotary positions of a `Sam3ViTLayer` (`precompute_positions`) in float64:
    transformers computes them in float32, where the positions of the global attention layers,
    scaled to the window by `window_size / grid`, are rounded."""
    config, window = layer.config, layer.window_size
    pair = lambda s: tuple(s) if isinstance(s, (list, tuple)) else (s, s)
    if window > 0:
        grid = (window, window)
    else:
        image, patch = pair(config.image_size), pair(config.patch_size)
        grid = (image[1] // patch[1], image[0] // patch[0])
    h, w = torch.meshgrid(torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij")
    positions = (torch.stack([w.flatten(), h.flatten()], dim=-1).double()
                 * (config.window_size / grid[1]))
    assert torch.allclose(positions, layer.position_ids.double(), rtol=1e-6, atol=0)
    return positions


def _interpolation_axis_taps_weights_float64(index, size, side, mode, align_corners,
                                             padding="border"):
    """`vision_utils._interpolation_axis_taps_weights` in float64."""
    index = index.to(torch.float64)
    if align_corners:
        src = index * (side - 1) / torch.clamp(size - 1, min=1)
    else:
        src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    assert mode == "bilinear", f"interpolation mode {mode} is not supported"
    offsets = torch.arange(0, 2, device=index.device)
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    weights = (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    if padding == "zeros":
        weights = weights * ((raw_taps >= 0) & (raw_taps <= side - 1))
    return taps, weights


@contextlib.contextmanager
def exact_float64_diffusers():
    """Compute the timestep embeddings, the rotary embeddings and the Euler steps of diffusers
    in the dtype of their inputs.

    diffusers computes them in float32 even in a float64 model or pipeline, which would limit
    the fixtures to float32 precision. Inputs of other dtypes are computed as diffusers does.
    So are diffusers' RMSNorms and those of Z-Image (see `_exact_float64_z_image`).
    """
    from diffusers.models import embeddings
    from diffusers.models.transformers import transformer_flux2
    from diffusers.schedulers import scheduling_flow_match_euler_discrete as euler

    def timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                           downscale_freq_shift=1, scale=1, max_period=10000):
        if timesteps.dtype != torch.float64:
            return originals[0](timesteps, embedding_dim, flip_sin_to_cos, downscale_freq_shift,
                                scale, max_period)
        half = embedding_dim // 2
        exponent = -np.log(max_period) * torch.arange(half, dtype=torch.float64) / (half - downscale_freq_shift)
        emb = scale * timesteps[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return torch.cat([emb[:, half:], emb[:, :half]], dim=-1) if flip_sin_to_cos else emb

    def pos_embed(self, ids):
        pos = ids.double()
        cos, sin = [], []
        for i, dim in enumerate(self.axes_dim):
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            angles = torch.outer(pos[..., i], freqs)
            cos.append(angles.cos().repeat_interleave(2, dim=1))
            sin.append(angles.sin().repeat_interleave(2, dim=1))
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)

    def rotary(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
        if x.dtype != torch.float64:
            return originals[2](x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim)
        assert use_real and use_real_unbind_dim == -1 and sequence_dim == 1
        cos, sin = (f[None, :, None, :].to(x.dtype) for f in freqs_cis)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        return x * cos + rotated * sin

    def step(self, model_output, timestep, sample, *args, return_dict=True, **kwargs):
        if sample.dtype != torch.float64:
            return originals[3](self, model_output, timestep, sample, *args,
                                return_dict=return_dict, **kwargs)
        if self.step_index is None:
            self._init_step_index(timestep)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self._step_index += 1
        prev_sample = sample + dt * model_output
        return (prev_sample,) if not return_dict else euler.FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    originals = (embeddings.get_timestep_embedding, transformer_flux2.Flux2PosEmbed.forward,
                 transformer_flux2.apply_rotary_emb, euler.FlowMatchEulerDiscreteScheduler.step)
    embeddings.get_timestep_embedding = timestep_embedding
    transformer_flux2.Flux2PosEmbed.forward = pos_embed
    transformer_flux2.apply_rotary_emb = rotary
    euler.FlowMatchEulerDiscreteScheduler.step = step
    try:
        with _exact_float64_z_image():
            yield
    finally:
        (embeddings.get_timestep_embedding, transformer_flux2.Flux2PosEmbed.forward,
         transformer_flux2.apply_rotary_emb, euler.FlowMatchEulerDiscreteScheduler.step) = originals


@contextlib.contextmanager
def _exact_float64_z_image():
    """Compute diffusers' `RMSNorm` and the timestep embeddings and rotary embeddings of the
    Z-Image transformer in float64 in a float64 model, which diffusers computes in float32.

    The timestep embeddings are computed in float64 from the timesteps of the model's dtype or
    of float32 (the pipeline's), and the rotary angles and their rotations of float64 tokens
    in float64."""
    from diffusers.models import normalization
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers import transformer_z_image as z_image

    def rms_norm(self, hidden_states):
        if hidden_states.dtype != torch.float64:
            return originals[0](self, hidden_states)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        return hidden_states

    def timestep_forward(self, t):
        if self.mlp[0].weight.dtype != torch.float64:
            return originals[1](self, t)
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float64) / half)
        args = t.double()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))

    def precompute_freqs_cis(dim, end, theta=256.0):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            angles = torch.outer(torch.arange(e, dtype=torch.float64), freqs)
            freqs_cis.append(torch.polar(torch.ones_like(angles), angles))
        return freqs_cis

    def attention(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                  freqs_cis=None):
        if hidden_states.dtype != torch.float64:
            return originals[3](self, attn, hidden_states, encoder_hidden_states,
                                attention_mask, freqs_cis)
        query, key, value = (proj(hidden_states).unflatten(-1, (attn.heads, -1))
                             for proj in (attn.to_q, attn.to_k, attn.to_v))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def rotate(x):
            x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)

        if freqs_cis is not None:
            query, key = rotate(query), rotate(key)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = dispatch_attention_fn(query, key, value, attn_mask=attention_mask,
                                              dropout_p=0.0, is_causal=False,
                                              backend=self._attention_backend,
                                              parallel_config=self._parallel_config)
        output = attn.to_out[0](hidden_states.flatten(2, 3))
        return attn.to_out[1](output) if len(attn.to_out) > 1 else output

    targets = [(normalization.RMSNorm, "forward", rms_norm),
               (z_image.TimestepEmbedder, "forward", timestep_forward),
               (z_image.RopeEmbedder, "precompute_freqs_cis", staticmethod(precompute_freqs_cis)),
               (z_image.ZSingleStreamAttnProcessor, "__call__", attention)]
    originals = [getattr(cls, name) for cls, name, _ in targets]
    saved = [cls.__dict__[name] for cls, name, _ in targets]
    for cls, name, patch in targets:
        setattr(cls, name, patch)
    try:
        yield
    finally:
        for (cls, name, _), original in zip(targets, saved):
            setattr(cls, name, original)


def state_dict_f32(model):
    return {k: (v.float() if v.is_floating_point() else v).detach().contiguous().clone()
            for k, v in model.state_dict().items()}


def classifier_name(model):
    try:
        return model.get_classifier(name_only=True)
    except (AttributeError, TypeError):
        return getattr(model, "pretrained_cfg", {}).get("classifier")


def versions():
    import diffusers
    import transformers
    return {"timm": timm.__version__, "transformers": transformers.__version__,
            "diffusers": diffusers.__version__, "torch": torch.__version__}


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
