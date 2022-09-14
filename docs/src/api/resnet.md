# ResNet-like models

This is the API reference for the ResNet inspired model structures present in Metalhead.jl.

## The higher-level model constructors

```@docs
ResNet
WideResNet
ResNeXt
SEResNet
SEResNeXt
Res2Net
Res2NeXt
```

## The core ResNet function

```@docs
Metalhead.resnet
```

## The ResNet model

```@docs
Metalhead.build_resnet
```

## Blocks and their builders

### Block functions

```@docs
Metalhead.basicblock
Metalhead.bottleneck
Metalhead.bottle2neck
```

### Downsampling functions

```@docs
Metalhead.downsample_identity
Metalhead.downsample_conv
Metalhead.downsample_pool
```

### Block builders

```@docs
Metalhead.basicblock_builder
Metalhead.bottleneck_builder
Metalhead.bottle2neck_builder
```

## Utility callbacks

```@docs
Metalhead.resnet_planes
Metalhead.resnet_stride
Metalhead.resnet_stem
```
