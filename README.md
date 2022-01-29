# Metalhead

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.github.io/Metalhead.jl/dev)
[![CI](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/FluxML/Metalhead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Metalhead.jl)

[Metalhead.jl](https://github.com/FluxML/Metalhead.jl) provides standard machine learning vision models for use with [Flux.jl](https://fluxml.ai). The architectures in this package make use of pure Flux layers, and they represent the best-practices for creating modules like residual blocks, inception blocks, etc. in Flux.

## Installation

```julia
]add Metalhead
```

## Available models

| Model Name                                       | Function                                                                                  | Pre-trained? |
|:-------------------------------------------------|:------------------------------------------------------------------------------------------|:------------:|
| [VGG](https://arxiv.org/abs/1409.1556)-11        | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-11 (w/ BN)                                   | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-13                                           | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-13 (w/ BN)                                   | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-16                                           | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-16 (w/ BN)                                   | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-19                                           | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| VGG-19 (w/ BN)                                   | [`VGG`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG.html)                 | N            |
| [ResNet](https://arxiv.org/abs/1512.03385)-18    | [`ResNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet.html)           | N            |
| ResNet-34                                        | [`ResNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet.html)           | N            |
| ResNet-50                                        | [`ResNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet.html)           | N            |
| ResNet-101                                       | [`ResNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet.html)           | N            |
| ResNet-152                                       | [`ResNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet.html)           | N            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)     | [`GoogLeNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.GoogLeNet.html)     | N            |
| [Inception-v3](https://arxiv.org/abs/1512.00567) | [`Inception3`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.Inception3.html)   | N            |
| [SqueezeNet](https://arxiv.org/abs/1602.07360)   | [`SqueezeNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.SqueezeNet.html)   | N            |
| [DenseNet](https://arxiv.org/abs/1608.06993)-121 | [`DenseNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet.html)       | N            |
| DenseNet-161                                     | [`DenseNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet.html)       | N            |
| DenseNet-169                                     | [`DenseNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet.html)       | N            |
| DenseNet-201                                     | [`DenseNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet.html)       | N            |
| [ResNeXt](https://arxiv.org/abs/1611.05431)-50   | [`ResNeXt`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNeXt.html)          | N            |
| ResNeXt-101                                      | [`ResNeXt`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNeXt.html)          | N            |
| ResNeXt-152                                      | [`ResNeXt`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNeXt.html)          | N            |
| [MobileNetv2](https://arxiv.org/abs/1801.04381)  | [`MobileNetv2`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.MobileNetv2.html)  | N            |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)  | [`MobileNetv3`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.MobileNetv3.html)  | N            |


## Getting Started

You can find the Metalhead.jl getting started guide [here](# "Quickstart").