# Using the ResNet model family in Metalhead.jl

ResNets are one of the most common convolutional neural network (CNN) models used today. Originally proposed by He et al. in [**Deep Residual Learning for Image Recognition**](https://arxiv.org/abs/1512.03385), they use a residual structure to learn identity mappings that strengthens gradient propagation, thereby helping to prevent the vanishing gradient problem and allow the advent of truly deep neural networks as used today.

Many variants on the original ResNet structure have since become widely used such as [Wide-ResNet](https://arxiv.org/abs/1605.07146), [ResNeXt](https://arxiv.org/abs/1611.05431v2), [SE-ResNet](https://arxiv.org/abs/1709.01507) and [Res2Net](https://www.notion.so/ResNet-user-guide-b4c09e5bb5ae41328165a3f160a104f6). Apart from suggesting modifications to the structure of the residual block, papers have also suggested modifying the stem of the network, adding newer regularisation options in the form of stochastic depth and DropBlock, and changing the downsampling path for the blocks to improve performance.

Metalhead provides an extensible, hackable yet powerful interface for working with ResNets that provides built-in toggles for commonly used options in papers and other deep learning libraries, while also allowing the user to build custom model structures if they want very easily.

## Pre-trained models

Metalhead provides a variety of pretrained models in the ResNet family to allow users to get started quickly with tasks like transfer learning. Pretrained models for [`ResNet`](@ref) with depth 18, 34, 50, 101 and 152 is supported, as is [`WideResNet`](@ref) with depths 50 and 101. [`ResNeXt`](@ref) also supports some configurations of pretrained models - to know more, check out the documentation for the model.

This is as easy as setting the `pretrain` keyword to `true` when constructing the model. For example, to load a pretrained `ResNet` with depth 50, you can do the following:

```julia
using Metalhead

model = ResNet(50; pretrain=true)
```

To check out more about using pretrained models, check out the [pretrained models guide](@ref pretrained).

## The mid-level function

Metalhead also provides a function for users looking to customise the ResNet family of models further. This function is named [`Metalhead.resnet`](@ref) and has a detailed docstring that describes all the various customisation options. You may want to open the above link in another tab, because we're going to be referring to it extensively to build a ResNet model of our liking.

First, let's take a peek at how we would write the vanilla ResNet-18 model using this function. We know from the docstring that we want to use `Metalhead.basicblock` for the block, since the paper uses bottleneck blocks for depths 50 and above. We also know that the number of block repeats in each stage of the model as per the paper - 2 for each. For all other options, the default values work well. So we can write the ResNet-18 model as follows:

```julia
resnet18 = Metalhead.resnet(Metalhead.basicblock, [2, 2, 2, 2])
```

What if we want to customise the number of output classes? That's easy; the model has several keyword arguments, one of which allows this. The docstring tells us that it is `nclasses`, and so we can write:

```julia
resnet18 = Metalhead.resnet(Metalhead.basicblock, [2, 2, 2, 2]; nclasses = 10)
```

Let's try customising this further. Say I want to make a ResNet-50-like model, but with [`StochasticDepth`](https://arxiv.org/abs/1603.09382) to provide even more regularisation, and also a custom pooling layer such as `AdaptiveMeanMaxPool`. Both of these options are provided by Metalhead out of the box, and so we can write:

```julia
using Metalhead: Layers # AdaptiveMeanMaxPool is in the Layers module in Metalhead

custom_resnet = Metalhead.resnet(Metalhead.bottleneck, [3, 4, 6, 3];
                                 pool_layer = Layers.AdaptiveMeanMaxPool((1, 1)),
                                 stochastic_depth_prob = 0.2)
```

To make this a ResNeXt-like model, all we need to do is configure the cardinality and the 
base width:

```julia
custom_resnet = Metalhead.resnet(Metalhead.bottleneck, [3, 4, 6, 3];
                                 cardinality = 32, base_width = 4,
                                 pool_layer = Layers.AdaptiveMeanMaxPool((1, 1)),
                                 stochastic_depth_prob = 0.2)
```

And we have a custom model, built with minimal effort! The documentation for `Metalhead.resnet` has been written with extensive care and in as much detail as possible to facilitate user ease. However, if you find anything difficult to understand, feel free to open an issue and we will be happy to help you out, and to improve the documentation where necessary.
