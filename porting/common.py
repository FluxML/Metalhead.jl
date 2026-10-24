"""Shared helpers for the Metalhead porting scripts."""
import contextlib
import copy
import functools
import importlib
import inspect
import json
import math
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
            elif t.ndim <= 1:
                uniform(t, 0.5, 1.5)
            else:
                normal(t, 0.5)


def exact_float64(model):
    """Compute the RMSNorms, rotary tables and routers of a transformers model in its own dtype.

    transformers computes them in float32 even in a float64 model, which would limit the
    fixtures to float32 precision. In a float32 model the results are unchanged but for the
    rounding of the rotary frequencies, which are computed in float64 here.
    """
    for m in model.modules():
        name = type(m).__name__
        if name.endswith("RMSNorm"):
            forward = _gemma_rmsnorm_forward if name.startswith("Gemma") else _rmsnorm_forward
            m.forward = types.MethodType(forward, m)
        elif name.startswith("GraniteMoe") and name.endswith("TopKRouter"):
            m.forward = types.MethodType(_granitemoe_router_forward, m)
        elif name.endswith("RotaryEmbedding"):
            if isinstance(m.rope_type, dict):  # a table for each layer type (Gemma 3)
                m.exact_inv_freq = {t: _exact_inv_freq(m.config.rope_parameters[t],
                                                       getattr(m, f"{t}_inv_freq"))
                                    for t in m.rope_type}
                m.forward = types.MethodType(_layer_type_rotary_forward, m)
            else:
                m.exact_inv_freq = _exact_inv_freq(m.config.rope_parameters, m.inv_freq)
                m.forward = types.MethodType(_rotary_forward, m)


def _rmsnorm_forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (x * torch.rsqrt(variance + self.variance_epsilon))


def _granitemoe_router_forward(self, hidden_states):
    router_logits = torch.nn.functional.linear(hidden_states, self.weight)
    top_k_logits, top_k_index = router_logits.topk(self.top_k, dim=-1)
    return top_k_index, torch.softmax(top_k_logits, dim=-1), router_logits


def _gemma_rmsnorm_forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + self.eps)) * (1.0 + self.weight)


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
def _layer_type_rotary_forward(self, x, position_ids, layer_type):
    return _rotary_tables(x, position_ids, self.exact_inv_freq[layer_type],
                          getattr(self, f"{layer_type}_attention_scaling"))


def _rotary_tables(x, position_ids, inv_freq, attention_scaling):
    freqs = position_ids[..., None].double() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return ((emb.cos() * attention_scaling).to(x.dtype),
            (emb.sin() * attention_scaling).to(x.dtype))


@contextlib.contextmanager
def exact_float64_diffusers():
    """Compute the timestep embeddings, the rotary embeddings and the Euler steps of diffusers
    in the dtype of their inputs.

    diffusers computes them in float32 even in a float64 model or pipeline, which would limit
    the fixtures to float32 precision. Inputs of other dtypes are computed as diffusers does.
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
        yield
    finally:
        (embeddings.get_timestep_embedding, transformer_flux2.Flux2PosEmbed.forward,
         transformer_flux2.apply_rotary_emb, euler.FlowMatchEulerDiscreteScheduler.step) = originals


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
