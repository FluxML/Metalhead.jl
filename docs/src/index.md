```@meta
CurrentModule = Metalhead
```

# Metalhead

[Metalhead.jl](https://github.com/FluxML/Metalhead.jl) provides standard machine learning vision models for use with [Flux.jl](https://fluxml.ai). The architectures in this package make use of pure Flux layers, and they represent the best-practices for creating modules like residual blocks, inception blocks, etc. in Flux. Metalhead also provides some building blocks for more complex models in the Layers module.

## Installation

```julia
julia> ]add Metalhead
```

## Available models

| Model Name                                             | Function                    | Pre-trained? |
|:-------------------------------------------------------|:----------------------------|:------------:|
| [VGG](https://arxiv.org/abs/1409.1556)                 | [`VGG`](@ref)               | Y (w/o BN)   |
| [ResNet](https://arxiv.org/abs/1512.03385)             | [`ResNet`](@ref)            | Y            |
| [WideResNet](https://arxiv.org/abs/1605.07146)         | [`WideResNet`](@ref)        | Y            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)           | [`GoogLeNet`](@ref)         | N            |
| [Inception-v3](https://arxiv.org/abs/1512.00567)       | [`Inceptionv3`](@ref)       | N            |
| [Inception-v4](https://arxiv.org/abs/1602.07261)       | [`Inceptionv4`](@ref)       | N            |
| [InceptionResNet-v2](https://arxiv.org/abs/1602.07261) | [`Inceptionv3`](@ref)       | N            |
| [SqueezeNet](https://arxiv.org/abs/1602.07360)         | [`SqueezeNet`](@ref)        | Y            |
| [DenseNet](https://arxiv.org/abs/1608.06993)           | [`DenseNet`](@ref)          | N            |
| [ResNeXt](https://arxiv.org/abs/1611.05431)            | [`ResNeXt`](@ref)           | Y            |
| [MobileNetv1](https://arxiv.org/abs/1704.04861)        | [`MobileNetv1`](@ref)       | N            |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)        | [`MobileNetv2`](@ref)       | N            |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)        | [`MobileNetv3`](@ref)       | N            |
| [EfficientNet](https://arxiv.org/abs/1905.11946)       | [`EfficientNet`](@ref)      | N            |
| [MLPMixer](https://arxiv.org/pdf/2105.01601)           | [`MLPMixer`](@ref)          | N            |
| [ResMLP](https://arxiv.org/abs/2105.03404)             | [`ResMLP`](@ref)            | N            |
| [gMLP](https://arxiv.org/abs/2105.08050)               | [`gMLP`](@ref)              | N            |
| [ViT](https://arxiv.org/abs/2010.11929)                | [`ViT`](@ref)               | N            |
| [ConvNeXt](https://arxiv.org/abs/2201.03545)           | [`ConvNeXt`](@ref)          | N            |
| [ConvMixer](https://arxiv.org/abs/2201.09792)          | [`ConvMixer`](@ref)         | N            |
| [UNet](https://arxiv.org/abs/1505.04597v1)             | [`UNet`](@ref)              | N            |

To contribute new models, see our [contributing docs](@ref Contributing-to-Metalhead.jl).

## Getting Started

You can find the Metalhead.jl getting started guide [here](@ref Quickstart).
