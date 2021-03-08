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
| VGG-11         | [`vgg11`](#)       | N            |
| VGG-11 (w/ BN) | [`vgg11bn`](#)     | N            |
| VGG-13         | [`vgg13`](#)       | N            |
| VGG-13 (w/ BN) | [`vgg13bn`](#)     | N            |
| VGG-16         | [`vgg16`](#)       | N            |
| VGG-16 (w/ BN) | [`vgg16bn`](#)     | N            |
| VGG-19         | [`vgg19`](#)       | Y            |
| VGG-19 (w/ BN) | [`vgg19bn`](#)     | N            |
| ResNet-18      | [`resnet18`](#)    | N            |
| ResNet-34      | [`resnet34`](#)    | N            |
| ResNet-50      | [`resnet50`](#)    | Y            |
| ResNet-101     | [`resnet101`](#)   | N            |
| ResNet-152     | [`resnet152`](#)   | N            |
| GoogLeNet      | [`googlenet`](#)   | Y            |
| Inception-v3   | [`inception3`](#)  | N            |
| SqueezeNet     | [`squeezenet`](#)  | Y            |
| DenseNet-121   | [`densenet121`](#) | Y            |
| DenseNet-161   | [`densenet161`](#) | N            |
| DenseNet-169   | [`densenet169`](#) | N            |
| DenseNet-201   | [`densenet201`](#) | N            |