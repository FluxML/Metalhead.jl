# A guide to getting started with Metalhead

Metalhead.jl is a library written in Flux.jl that is a collection of image models, layers and utilities for deep learning in computer vision.

## Pre-trained models

In Metalhead.jl, camel-cased functions mimicking the naming style followed in the paper such as [`ResNet`](@ref) or [`ResNeXt`](@ref) are considered the "higher" level API for models. These are the functions that end-users who do not want to experiment much with model architectures should use. These models also support the option for loading pre-trained weights from ImageNet.

!!! note

	Metalhead is still under active development and thus not all models have pre-trained weights supported. While we are working on expanding the footprint of the pre-trained models, if you would like to help contribute model weights yourself, please check out the [contributing guide](@ref contributing) guide.

To use a pre-trained model, just instantiate the model with the `pretrain` keyword argument set to `true`:

```julia
using Metalhead
  
model = ResNet(18; pretrain = true);
```

Refer to the pretraining guide for more details on how to use pre-trained models.

## More model configuration options