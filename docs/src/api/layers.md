# Layers

Metalhead also defines a module called `Layers` which contains some custom layers that are used to configure the models in Metalhead. These layers are not available in Flux at present. To use the functions defined in the `Layers` module, you need to import it.

```julia
using Metalhead: Layers
```

This page contains the API reference for the `Layers` module.

!!! warning

    The `Layers` module is still a work in progress. While we will endeavour to keep the API stable, we cannot guarantee that it will not change in the future. If you find any of the functions in this module do not work as expected, please open an issue on GitHub.

## Convolution + BatchNorm layers

```@docs
Metalhead.Layers.conv_norm
Metalhead.Layers.basic_conv_bn
```

## Convolution-related custom blocks

These blocks are designed to be used in convolutional neural networks. Most of these are used in the MobileNet and EfficientNet family of models, but they also feature in "fancier" versions of well known-models like ResNet (SE-ResNet).

```@docs
Metalhead.Layers.dwsep_conv_norm
Metalhead.Layers.mbconv
Metalhead.Layers.fused_mbconv
Metalhead.Layers.squeeze_excite
Metalhead.Layers.effective_squeeze_excite
```

## Normalisation, Dropout and Pooling layers

Metalhead provides various custom layers for normalisation, dropout and pooling which have been used to additionally customise various models.

### Normalisation layers

```@docs
Metalhead.Layers.ChannelLayerNorm
Metalhead.Layers.LayerNormV2
Metalhead.Layers.LayerScale
```

### Dropout layers

```@docs
Metalhead.Layers.DropBlock
Metalhead.Layers.dropblock
Metalhead.Layers.StochasticDepth
```

### Pooling layers

```@docs
Metalhead.Layers.AdaptiveMeanMaxPool
```

## Classifier creation

Metalhead provides a function to create a classifier for neural network models that is quite flexible, and is used by the library extensively to create the classifier "head" for networks.

```@docs
Metalhead.Layers.create_classifier
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

## Utilities for layers

These are some miscellaneous utilities present in the `Layers` module, and are used with other custom/inbuilt layers to make certain common operations in neural networks easier.

```@docs
Metalhead.Layers.inputscale
Metalhead.Layers.actadd
Metalhead.Layers.addact
Metalhead.Layers.cat_channels
Metalhead.Layers.flatten_chains
Metalhead.Layers.linear_scheduler
Metalhead.Layers.swapdims
```
