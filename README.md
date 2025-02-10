# Metalhead

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.github.io/Metalhead.jl/dev)
[![CI](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/FluxML/Metalhead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Metalhead.jl)

[Metalhead.jl](https://github.com/FluxML/Metalhead.jl) provides standard machine learning vision models for use with [Flux.jl](https://fluxml.ai). The architectures in this package make use of pure Flux layers, and they represent the best-practices for creating modules like residual blocks, inception blocks, etc. in Flux. Metalhead also provides some building blocks for more complex models in the Layers module.

## Installation

```julia
julia> ]add Metalhead
```

## Getting Started

You can find the Metalhead.jl getting started guide [here](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/).

## Available models

To contribute new models, see our [contributing docs](https://fluxml.ai/Metalhead.jl/dev/contributing/).

### Image Classification

| Model Name                                       | Constructor                                                                                       | Pre-trained? |
|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------|:------------:|
| [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)    | [`AlexNet`](https://fluxml.ai/Metalhead.jl/dev/api/others/#Metalhead.AlexNet)       | N            |
| [ConvMixer](https://arxiv.org/abs/2201.09792)    | [`ConvMixer`](https://fluxml.ai/Metalhead.jl/dev/api/hybrid/#Metalhead.ConvMixer)       | N            |
| [ConvNeXt](https://arxiv.org/abs/2201.03545)     | [`ConvNeXt`](https://fluxml.ai/Metalhead.jl/dev/api/hybrid/#Metalhead.ConvNeXt)         | N            |
| [DenseNet](https://arxiv.org/abs/1608.06993)     | [`DenseNet`](https://fluxml.ai/Metalhead.jl/dev/api/densenet/#Metalhead.DenseNet)         | N            |
| [EfficientNet](https://arxiv.org/abs/1905.11946) | [`EfficientNet`](https://fluxml.ai/Metalhead.jl/dev/api/efficientnet/#Metalhead.EfficientNet) | N            |
| [EfficientNetv2](https://arxiv.org/abs/2104.00298) | [`EfficientNetv2`](https://fluxml.ai/Metalhead.jl/dev/api/efficientnet/#Metalhead.EfficientNetv2) | N            |
| [gMLP](https://arxiv.org/abs/2105.08050)         | [`gMLP`](https://fluxml.ai/Metalhead.jl/dev/api/mixers/#Metalhead.gMLP)                 | N            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)     | [`GoogLeNet`](https://fluxml.ai/Metalhead.jl/dev/api/inception/#Metalhead.GoogLeNet)       | N            |
| [Inception-v3](https://arxiv.org/abs/1512.00567) | [`Inceptionv3`](https://fluxml.ai/Metalhead.jl/dev/api/inception/#Metalhead.Inceptionv3)   | N            |
| [Inception-v4](https://arxiv.org/abs/1602.07261) | [`Inceptionv4`](https://fluxml.ai/Metalhead.jl/dev/api/inception/#Metalhead.Inceptionv4)   | N            |
| [InceptionResNet-v2](https://arxiv.org/abs/1602.07261) | [`InceptionResNetv2`](https://fluxml.ai/Metalhead.jl/dev/api/inception/#Metalhead.InceptionResNetv2) | N            |
| [MLPMixer](https://arxiv.org/pdf/2105.01601)     | [`MLPMixer`](https://fluxml.ai/Metalhead.jl/dev/api/mixers/#Metalhead.MLPMixer)         | N            |
| [MobileNetv1](https://arxiv.org/abs/1704.04861)  | [`MobileNetv1`](https://fluxml.ai/Metalhead.jl/dev/api/mobilenet/#Metalhead.MobileNetv1)   | N            |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)  | [`MobileNetv2`](https://fluxml.ai/Metalhead.jl/dev/api/mobilenet/#Metalhead.MobileNetv2)   | Y            |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)  | [`MobileNetv3`](https://fluxml.ai/Metalhead.jl/dev/api/mobilenet/#Metalhead.MobileNetv3)   | Y            |
| [MNASNet](https://arxiv.org/abs/1807.11626)       | [`MNASNet`](https://fluxml.ai/Metalhead.jl/dev/api/efficientnet/#Metalhead.MNASNet)   | N            |
| [ResMLP](https://arxiv.org/abs/2105.03404)       | [`ResMLP`](https://fluxml.ai/Metalhead.jl/dev/api/mixers/#Metalhead.ResMLP)                    | N            |
| [ResNet](https://arxiv.org/abs/1512.03385)       | [`ResNet`](https://fluxml.ai/Metalhead.jl/dev/api/resnet/#Metalhead.ResNet)             | Y            |
| [ResNeXt](https://arxiv.org/abs/1611.05431)      | [`ResNeXt`](https://fluxml.ai/Metalhead.jl/dev/api/resnet/#Metalhead.ResNeXt)           | Y            |
| [SqueezeNet](https://arxiv.org/abs/1602.07360)   | [`SqueezeNet`](https://fluxml.ai/Metalhead.jl/dev/api/others/#Metalhead.SqueezeNet)     | Y            |
| [Xception](https://arxiv.org/abs/1610.02357) | [`Xception`](https://fluxml.ai/Metalhead.jl/dev/api/inception/#Metalhead.Xception)                 | N            |
| [WideResNet](https://arxiv.org/abs/1605.07146)   | [`WideResNet`](https://fluxml.ai/Metalhead.jl/dev/api/resnet/#Metalhead.WideResNet)     | Y            |
| [VGG](https://arxiv.org/abs/1409.1556)           | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/api/others/#Metalhead.VGG)                   | Y            |
| [Vision Transformer](https://arxiv.org/abs/2010.11929) | [`ViT`](https://fluxml.ai/Metalhead.jl/dev/api/vit/#Metalhead.ViT)             | Y            |

### Other Models

| Model Name                                       | Constructor                                                                                       | Pre-trained? |
|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------|:------------:|
| [UNet](https://arxiv.org/abs/1505.04597)         | [`UNet`](https://fluxml.ai/Metalhead.jl/dev/api/others/#Metalhead.UNet)                         | N            |
