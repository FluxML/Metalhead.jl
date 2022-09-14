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

For users who want to use more options for model configuration, Metalhead provides a "mid-level" API for models. The model functions that are in lowercase such as [`resnet`](@ref) or [`mobilenetv3`](@ref) are the "lower" level API for models. These are the functions that end-users who want to experiment with model architectures should use. These models do not support the option for loading pre-trained weights from ImageNet out of the box.

To use any of these models, check out the docstrings for the model functions. Note that these functions typically require more configuration options to be passed in, but offer a lot more flexibility in terms of model architecture.

##