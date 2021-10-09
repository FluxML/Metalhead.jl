# Metalhead

[![Build Status](https://travis-ci.org/FluxML/Metalhead.jl.svg?branch=master)](https://travis-ci.org/FluxML/Metalhead.jl) [![Coverage](https://codecov.io/gh/FluxML/Metalhead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Metalhead.jl)

[Metalhead.jl](https://github.com/FluxML/Metalhead.jl) provides standard machine learning vision models for use with [Flux.jl](https://fluxml.ai). The architectures in this package make use of pure Flux layers, and they represent the best-practices for creating modules like residual blocks, inception blocks, etc. in Flux.

## Installation

```julia
]add Metalhead
```

## Available models

| Model Name     | Function           | Pre-trained? |
|:---------------|:-------------------|:------------:|
| [VGG](https://arxiv.org/abs/1409.1556)-11          | [`VGG11`](src/vgg.jl)             | N            |
| VGG-11 (w/ BN) | [`VGG11`](src/vgg.jl)             | N            |
| VGG-13         | [`VGG13`](src/vgg.jl)             | N            |
| VGG-13 (w/ BN) | [`VGG13`](src/vgg.jl)             | N            |
| VGG-16         | [`VGG16`](src/vgg.jl)             | N            |
| VGG-16 (w/ BN) | [`VGG16`](src/vgg.jl)             | N            |
| VGG-19         | [`VGG19`](src/vgg.jl)             | Y            |
| VGG-19 (w/ BN) | [`VGG19`](src/vgg.jl)             | N            |
| [ResNet](https://arxiv.org/abs/1512.03385)-18      | [`ResNet18`](src/resnet.jl)       | N            |
| ResNet-34      | [`ResNet34`](src/resnet.jl)       | N            |
| ResNet-50      | [`ResNet50`](src/resnet.jl)       | Y            |
| ResNet-101     | [`ResNet101`](src/resnet.jl)      | N            |
| ResNet-152     | [`ResNet152`](src/resnet.jl)      | N            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)      | [`GoogLeNet`](src/googlenet.jl)   | Y            |
| [Inception-v3](https://arxiv.org/abs/1512.00567)   | [`Inception3`](src/inception.jl)  | N            |
| [SqueezeNet](https://arxiv.org/abs/1602.07360)     | [`SqueezeNet`](src/squeezenet.jl) | Y            |
| [DenseNet](https://arxiv.org/abs/1608.06993)-121   | [`DenseNet121`](src/densenet.jl)  | Y            |
| DenseNet-161   | [`DenseNet161`](src/densenet.jl)  | N            |
| DenseNet-169   | [`DenseNet169`](src/densenet.jl)  | N            |
| DenseNet-201   | [`DenseNet201`](src/densenet.jl)  | N            |
