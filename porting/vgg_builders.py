"""Builders for the VGG fixtures: `timm.create_model` can't take `cfg`, and the class
`norm_layer` of the `_bn` variants can't be written in a JSON spec."""
from torch import nn
from timm.models.vgg import VGG


def vgg_bn(**kwargs):
    return VGG(norm_layer=nn.BatchNorm2d, **kwargs)
