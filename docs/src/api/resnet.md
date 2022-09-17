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

## The mid-level function

```@docs
Metalhead.resnet
```

## Lower-level functions and builders

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

### Generic ResNet model builder

```@docs
Metalhead.build_resnet
```

## Utility callbacks

```@docs
Metalhead.resnet_planes
Metalhead.resnet_stride
Metalhead.resnet_stem
```
