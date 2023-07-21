# [An introduction to the `Layers` module in Metalhead.jl](@id layers-intro)

Since v0.8, Metalhead.jl exports a `Layers` module that contains a number of useful layers and utilities for building neural networks. This guide will walk you through the most commonly used layers and utilities present in the `Layers` module, and how to use them. It also contains some examples of how these layers are used in Metalhead.jl as well as a comprehensive API reference.

!!! warning

    The `Layers` module is still a work in progress. While we will endeavour to keep the API stable, we cannot guarantee that it will not change in the future. In particular, the API may change significantly between major versions of Metalhead.jl. If you find any of the functions in this module do not work as expected, please open an issue on GitHub.

First, however, you want to make sure that the `Layers` module is loaded, and that the functions and types are available in your current scope. You can do this by running the following code:

```julia
using Metalhead
using Metalhead.Layers
```

## Convolution + Normalisation: the `conv_norm` layer

One of the most common patterns in modern neural networks is to have a convolutional layer followed by a normalisation layer. Most major deep learning libraries have a way to combine these two layers into a single layer. In Metalhead.jl, this is done with the [`Metalhead.Layers.conv_norm`](@ref) layer. The function signature for this is given below:

```@docs
Metalhead.Layers.conv_norm
```

To know more about the exact details of each of these parameters, you can refer to the documentation for this function. For now, we will focus on some common use cases. For example, if you want to create a convolutional layer with a kernel size of 3x3, with 32 input channels and 64 output channels, along with a `BatchNorm` layer, you can do the following:

```julia
conv_norm((3, 3), 32, 64)
```

This returns a `Vector` with the desired layers. To use it in a model, the user should splat it into a Chain. For example:

```julia
Chain(Dense(3, 32), conv_norm((3, 3), 32, 64)..., Dense(64, 10))
```

The default activation function for `conv_norm` is `relu`, and the default normalisation layer is `BatchNorm`. To use a different activation function, you can just pass it in as a positional argument. For example, to use a `sigmoid` activation function:

```julia
conv_norm((3, 3), 32, 64, sigmoid)
```

Let's try something else. Suppose you want to use a `GroupNorm` layer instead of a `BatchNorm` layer. Note that `norm_layer` is a keyword argument in the function signature of `conv_norm` as shown above. Then we can write:

```julia
conv_norm((3, 3), 32, 64; norm_layer = GroupNorm)
```

What if you want to change certain specific parameters of the `norm_layer`? For example, what if you want to change the number of groups in the `GroupNorm` layer?

```julia
# defining the norm layer
norm_layer = planes -> GroupNorm(planes, 4)
# passing it to the conv_norm layer
conv_norm((3, 3), 32, 64; norm_layer = norm_layer)
```

One of Julia's features is that functions are first-class objects, and can be passed around as arguments to other functions. Here, we have create an [**anonymous function**](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions-1) that takes in the number of planes as an argument, and returns a `GroupNorm` layer with 4 groups. This is then passed to the `norm_layer` keyword argument of the `conv_norm` layer. Using anonymous functions allows us to configure the layers in a very flexible manner, and this is a common pattern in Metalhead.jl.

Let's take a slightly more complicated example. TensorFlow uses different defaults for its normalisation layers. In particular, it uses an `epsilon` value of `1e-3` for `BatchNorm` layers. If you want to use the same defaults as TensorFlow, you can do the following:

```julia
# note that 1e-3 is not a Float32 and Flux is optimized for Float32, so we use 1.0f-3
conv_norm((3, 3), 32, 64; norm_layer = planes -> BatchNorm(planes, eps = 1.0f-3))
```

which, incidentally, is very similar to the code Metalhead uses internally for the [`Metalhead.Layers.basic_conv_bn`](@ref) layer that is used in the Inception family of models.

```@docs
Metalhead.Layers.basic_conv_bn
```

## Normalisation layers

The `Layers` module provides some custom normalisation functions that are not present in Flux.
 
```@docs
Metalhead.Layers.LayerScale
Metalhead.Layers.LayerNormV2
Metalhead.Layers.ChannelLayerNorm
```

There is also a utility function, `prenorm`, which applies a normalisation layer before a given block and simply returns a `Chain` with the normalisation layer and the block. This is useful for creating Vision Transformers (ViT)-like models.

```@docs
Metalhead.Layers.prenorm
```

## Dropout layers

The `Layers` module provides two dropout-like layers not present in Flux:

```@docs
Metalhead.Layers.DropBlock
Metalhead.Layers.StochasticDepth
```

`DropBlock` also has a functional variant present in the `Layers` module:

```@docs
Metalhead.Layers.dropblock
```

Both `DropBlock` and `StochasticDepth` are used along with probability values that vary based on a linear schedule across the structure of the model (see the respective papers for more details). The `Layers` module provides a utility function to create such a schedule as well:

```@docs
Metalhead.Layers.linear_scheduler
```

The [`Metalhead.resnet`](@ref) function which powers the ResNet family of models in Metalhead.jl is configured to allow the use of both these layers. For examples, check out the guide for using the ResNet family in Metalhead [here](@ref resnet-guide). These layers can also be used by the user to construct other custom models.

## Pooling layers

The `Layers` module provides a [`Metalhead.Layers.AdaptiveMeanMaxPool`](@ref) layer, which is inspired by a similar layer present in [timm](https://github.com/huggingface/pytorch-image-models/blob/394e8145551191ae60f672556936314a20232a35/timm/layers/adaptive_avgmax_pool.py#L106). 

```@docs
Metalhead.Layers.AdaptiveMeanMaxPool
```

Many mid-level model functions in Metalhead.jl have been written to support passing custom pooling layers to them if applicable (either in the model itself or in the classifier head). For example, the [`Metalhead.resnet`](@ref) function supports this, and examples of this can be found in the guide for using the ResNet family in Metalhead [here](@ref resnet-guide).

## Classifier creation

Metalhead provides a function to create a classifier for neural network models that is quite flexible, and is used by the library extensively to create the classifier "head" for networks.
This function is called [`Metalhead.Layers.create_classifier`](@ref) and is documented below:

```@docs
Metalhead.Layers.create_classifier
```

Due to the power of multiple dispatch in Julia, the above function can be called with two different signatures - one of which creates a classifier with no hidden layers, and the other which creates a classifier with a single hidden layer. The function signature for both is documented above, and the user can choose the one that is most convenient for them. Both are used in Metalhead.jl - the latter is used in MobileNetv3, and the former is used almost everywhere else.
