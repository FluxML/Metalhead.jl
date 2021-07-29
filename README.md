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
| VGG-11         | [`VGG11`](#)       | N            |
| VGG-11 (w/ BN) | [`VGG11`](#)       | N            |
| VGG-13         | [`VGG13`](#)       | N            |
| VGG-13 (w/ BN) | [`VGG13`](#)       | N            |
| VGG-16         | [`VGG16`](#)       | N            |
| VGG-16 (w/ BN) | [`VGG16`](#)       | N            |
| VGG-19         | [`VGG19`](#)       | Y            |
| VGG-19 (w/ BN) | [`VGG19`](#)       | N            |
| ResNet-18      | [`ResNet18`](#)    | N            |
| ResNet-34      | [`ResNet34`](#)    | N            |
| ResNet-50      | [`ResNet50`](#)    | Y            |
| ResNet-101     | [`ResNet101`](#)   | N            |
| ResNet-152     | [`ResNet152`](#)   | N            |
| GoogLeNet      | [`GoogLeNet`](#)   | Y            |
| Inception-v3   | [`Inception3`](#)  | N            |
| SqueezeNet     | [`SqueezeNet`](#)  | Y            |
| DenseNet-121   | [`DenseNet121`](#) | Y            |
| DenseNet-161   | [`DenseNet161`](#) | N            |
| DenseNet-169   | [`DenseNet169`](#) | N            |
| DenseNet-201   | [`DenseNet201`](#) | N            |
