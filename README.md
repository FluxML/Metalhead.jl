# Metalhead

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.github.io/Metalhead.jl/dev)
[![CI](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/FluxML/Metalhead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Metalhead.jl)

[Metalhead.jl](https://github.com/FluxML/Metalhead.jl) provides standard machine learning vision models for use with [Flux.jl](https://fluxml.ai). The architectures in this package make use of pure Flux layers, and they represent the best-practices for creating modules like residual blocks, inception blocks, etc. in Flux. Metalhead also provides some building blocks for more complex models in the Layers module.

## Installation

```julia
julia> ]add Metalhead
```

## Available models

| Model Name                                       | Function                                                                                       | Pre-trained? |
|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------|:------------:|
| [VGG](https://arxiv.org/abs/1409.1556)           | [`VGG`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.VGG)                   | Y (w/o BN)   |
| [ResNet](https://arxiv.org/abs/1512.03385)       | [`ResNet`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.ResNet)             | Y            |
| [WideResNet](https://arxiv.org/abs/1605.07146)   | [`WideResNet`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.WideResNet)     | Y            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)     | [`GoogLeNet`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.GoogLeNet)       | N            |
| [Inception-v3](https://arxiv.org/abs/1512.00567) | [`Inceptionv3`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.Inceptionv3)   | N            |
| [Inception-v4](https://arxiv.org/abs/1602.07261) | [`Inceptionv4`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.Inceptionv4)   | N            |
| [InceptionResNet-v2](https://arxiv.org/abs/1602.07261) | [`Inceptionv3`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.InceptionResNetv2) | N            |
| [SqueezeNet](https://arxiv.org/abs/1602.07360)   | [`SqueezeNet`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.SqueezeNet)     | Y            |
| [DenseNet](https://arxiv.org/abs/1608.06993)     | [`DenseNet`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.DenseNet)         | N            |
| [ResNeXt](https://arxiv.org/abs/1611.05431)      | [`ResNeXt`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.ResNeXt)           | Y            |
| [MobileNetv1](https://arxiv.org/abs/1704.04861)  | [`MobileNetv1`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.MobileNetv1)   | N            |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)  | [`MobileNetv2`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.MobileNetv2)   | N            |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)  | [`MobileNetv3`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.MobileNetv3)   | N            |
| [EfficientNet](https://arxiv.org/abs/1905.11946) | [`EfficientNet`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.EfficientNet) | N            |
| [MLPMixer](https://arxiv.org/pdf/2105.01601)     | [`MLPMixer`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.MLPMixer)         | N            |
| [ResMLP](https://arxiv.org/abs/2105.03404)       | [`ResMLP`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.ResMLP)             | N            |
| [gMLP](https://arxiv.org/abs/2105.08050)         | [`gMLP`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.gMLP)                 | N            |
| [ViT](https://arxiv.org/abs/2010.11929)          | [`ViT`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.ViT)                   | N            |
| [ConvNeXt](https://arxiv.org/abs/2201.03545)     | [`ConvNeXt`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.ConvNeXt)         | N            |
| [ConvMixer](https://arxiv.org/abs/2201.09792)    | [`ConvMixer`](https://fluxml.ai/Metalhead.jl/stable/api/reference.html#Metalhead.ConvMixer)       | N            |

To contribute new models, see our [contributing docs](https://fluxml.ai/Metalhead.jl/dev/contributing/).

## Getting Started

You can find the Metalhead.jl getting started guide [here](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/).
