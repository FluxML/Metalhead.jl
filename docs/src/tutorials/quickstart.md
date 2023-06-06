# A guide to getting started with Metalhead

Metalhead.jl is a library written in Flux.jl that is a collection of image models, layers and utilities for deep learning in computer vision.

## Model architectures and pre-trained models

In Metalhead.jl, camel-cased functions mimicking the naming style followed in the paper such as [`ResNet`](@ref) or [`MobileNetv3`](@ref) are considered the "higher" level API for models. These are the functions that end-users who do not want to experiment much with model architectures should use. To use these models, simply call the function of the model:

```julia
using Metalhead

model = ResNet(18);
```

The API reference contains the documentation and options for each model function. These models also support the option for loading pre-trained weights from ImageNet.

!!! note

	Metalhead is still under active development and thus not all models have pre-trained weights supported. While we are working on expanding the footprint of the pre-trained models, if you would like to help contribute model weights yourself, please check out the [contributing guide](@ref contributing) guide.

To use a pre-trained model, just instantiate the model with the `pretrain` keyword argument set to `true`:

```julia
using Metalhead
  
model = ResNet(18; pretrain = true);
```

Refer to the [pretraining guide](@pretrained) for more details on how to use pre-trained models.

## More model configuration options

For users who want to use more options for model configuration, Metalhead provides a "mid-level" API for models. These are the model functions that are in lowercase such as [`resnet`](@ref) or [`mobilenetv3`](@ref). End-users who want to experiment with model architectures should use these functions. These models do not support the option for loading pre-trained weights from ImageNet out of the box, although one can always load weights explicitly using the `loadmodel!` function from Flux.

To use any of these models, check out the docstrings for the model functions (these are documented in the API reference). Note that these functions typically require more configuration options to be passed in, but offer a lot more flexibility in terms of model architecture. Metalhead defines as many default options as possible so as to make it easier for the user to pick and choose specific options to customise.

## Builders for the advanced user

For users who want the ability to customise their models as much as possible, Metalhead offers a powerful low-level interface. These are known as [**builders**](@ref builders) and allow the user to hack into the core of models and build them up as per their liking. Most users will not need to use builders since a large number of configuration options are exposed at the mid-level API. However, for package developers and users who want to build customised versions of their own models, the low-level API provides the customisability required while still reducing user code.
