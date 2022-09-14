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

## More configuration options