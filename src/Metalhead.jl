module Metalhead

using Flux
using Flux: outputsize, Zygote
using Functors
using BSON
using Artifacts, LazyArtifacts

import Functors

include("utilities.jl")

# CNN models
include("convnets/ConvNets.jl")
using .ConvNets
export  AlexNet,
        VGG, VGG11, VGG13, VGG16, VGG19,
        ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        GoogLeNet, Inception3, SqueezeNet,
        DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
        ResNeXt,
        MobileNetv2, MobileNetv3

# ViT models
include("vit-based/ViT.jl")
using .ViT
export  MLPMixer

end # module
