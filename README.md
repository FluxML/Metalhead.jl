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
| [VGG](https://arxiv.org/abs/1409.1556)-11        | [`VGG11`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG11.html)             | N            |
| VGG-11 (w/ BN)                                   | [`VGG11`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG11.html)             | N            |
| VGG-13                                           | [`VGG13`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG13.html)             | N            |
| VGG-13 (w/ BN)                                   | [`VGG13`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG13.html)             | N            |
| VGG-16                                           | [`VGG16`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG16.html)             | N            |
| VGG-16 (w/ BN)                                   | [`VGG16`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG16.html)             | N            |
| VGG-19                                           | [`VGG19`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG19.html)             | N            |
| VGG-19 (w/ BN)                                   | [`VGG19`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.VGG19.html)             | N            |
| [ResNet](https://arxiv.org/abs/1512.03385)-18    | [`ResNet18`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet18.html)       | N            |
| ResNet-34                                        | [`ResNet34`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet34.html)       | N            |
| ResNet-50                                        | [`ResNet50`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet50.html)       | N            |
| ResNet-101                                       | [`ResNet101`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet101.html)     | N            |
| ResNet-152                                       | [`ResNet152`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.ResNet152.html)     | N            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)     | [`GoogLeNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.GoogLeNet.html)     | N            |
| [Inception-v3](https://arxiv.org/abs/1512.00567) | [`Inception3`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.Inception3.html)   | N            |
| [SqueezeNet](https://arxiv.org/abs/1602.07360)   | [`SqueezeNet`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.SqueezeNet.html)   | N            |
| [DenseNet](https://arxiv.org/abs/1608.06993)-121 | [`DenseNet121`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet121.html) | N            |
| DenseNet-161                                     | [`DenseNet161`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet161.html) | N            |
| DenseNet-169                                     | [`DenseNet169`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet169.html) | N            |
| DenseNet-201                                     | [`DenseNet201`](https://fluxml.ai/Metalhead.jl/dev/docstrings/Metalhead.DenseNet201.html) | N            |

## Getting Started

You can find the Metalhead.jl getting started guide here: https://fluxml.ai/Metalhead.jl/dev/docs/tutorials/quickstart.html