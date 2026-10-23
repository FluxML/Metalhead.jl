"""Shared helpers for the Metalhead porting scripts."""
import importlib
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
    """Build the model described by a spec: `timm.create_model` or a named builder."""
    kwargs = dict(spec.get("kwargs", {}), **extra)
    if spec.get("builder"):
        module, fn = spec["builder"].split(":")
        return getattr(importlib.import_module(module), fn)(**kwargs)
    return timm.create_model(spec["architecture"], pretrained=False, **kwargs)


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
