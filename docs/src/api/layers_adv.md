# More advanced layers

This page contains the API reference for some more advanced layers present in the `Layers` module. These layers are used in Metalhead.jl to build more complex models, and can also be used by the user to build custom models. For a more basic introduction to the `Layers` module, please refer to the [introduction guide](@ref layers-intro) for the `Layers` module.

## Squeeze-and-excitation blocks

These are used in models like SE-ResNet and SE-ResNeXt, as well as in the design of inverted residual blocks used in the MobileNet and EfficientNet family of models.

```@docs
Metalhead.Layers.squeeze_excite
Metalhead.Layers.effective_squeeze_excite
```

## Inverted residual blocks

These blocks are designed to be used in the MobileNet and EfficientNet family of convolutional neural networks.

```@docs
Metalhead.Layers.dwsep_conv_norm
Metalhead.Layers.mbconv
Metalhead.Layers.fused_mbconv
```

## Vision transformer-related layers

The `Layers` module contains specific layers that are used to build vision transformer (ViT)-inspired models:

```@docs
Metalhead.Layers.MultiHeadSelfAttention
Metalhead.Layers.ClassTokens
Metalhead.Layers.ViPosEmbedding
Metalhead.Layers.PatchEmbedding
```

## MLPMixer-related blocks

Apart from this, the `Layers` module also contains certain blocks used in MLPMixer-style models:

```@docs
Metalhead.Layers.gated_mlp_block
Metalhead.Layers.mlp_block
```

## Miscellaneous utilities for layers

These are some miscellaneous utilities present in the `Layers` module, and are used with other custom/inbuilt layers to make certain common operations in neural networks easier.

```@docs
Metalhead.Layers.inputscale
Metalhead.Layers.actadd
Metalhead.Layers.addact
Metalhead.Layers.cat_channels
Metalhead.Layers.flatten_chains
Metalhead.Layers.swapdims
```
