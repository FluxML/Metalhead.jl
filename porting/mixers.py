"""Builders of the MLP-Mixer and ConvMixer fixtures.

The timm entry points fix `dim`, `depth`, `num_blocks`, `embed_dim`, ... so these builders take the
classes by name, like the Metalhead builders: see `mlp_mixer` in src/Timm/mlp_mixer.jl.
"""
from functools import partial

from torch import nn
from timm.layers import GatedMlp, GluMlp, Mlp
from timm.models import convmixer, mlp_mixer

BLOCKS = {"MixerBlock": mlp_mixer.MixerBlock, "ResBlock": mlp_mixer.ResBlock,
          "SpatialGatingBlock": mlp_mixer.SpatialGatingBlock}
MLPS = {"Mlp": Mlp, "GluMlp": GluMlp, "GatedMlp": GatedMlp}
NORMS = {"layernorm": partial(nn.LayerNorm, eps=1e-6), "affine": mlp_mixer.Affine}
ACTS = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}


def mixer(block_layer="MixerBlock", mlp_layer="Mlp", norm_layer="layernorm", act_layer="gelu",
          init_values=None, mlp_ratio=(0.5, 4.0), **kwargs):
    block = BLOCKS[block_layer]
    if init_values is not None:
        block = partial(block, init_values=init_values)
    model = mlp_mixer.MlpMixer(
        block_layer=block, mlp_layer=MLPS[mlp_layer], norm_layer=NORMS[norm_layer],
        act_layer=ACTS[act_layer],
        mlp_ratio=tuple(mlp_ratio) if isinstance(mlp_ratio, list) else mlp_ratio, **kwargs)
    model.pretrained_cfg = {"classifier": "head"}
    return model


def conv_mixer(act_layer="gelu", **kwargs):
    model = convmixer.ConvMixer(act_layer=ACTS[act_layer], **kwargs)
    model.pretrained_cfg = {"classifier": "head"}
    return model
