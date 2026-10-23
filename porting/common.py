"""Shared helpers for the Metalhead porting scripts."""
import contextlib
import copy
import functools
import importlib
import inspect
import json

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
    """
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
            elif t.ndim <= 1:
                uniform(t, 0.5, 1.5)
            else:
                normal(t, 0.5)


def state_dict_f32(model):
    return {k: (v.float() if v.is_floating_point() else v).detach().contiguous().clone()
            for k, v in model.state_dict().items()}


def classifier_name(model):
    try:
        return model.get_classifier(name_only=True)
    except (AttributeError, TypeError):
        return getattr(model, "pretrained_cfg", {}).get("classifier")


def versions():
    return {"timm": timm.__version__, "torch": torch.__version__}


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
