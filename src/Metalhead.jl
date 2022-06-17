module Metalhead

using Flux
using Flux: Zygote, outputsize
using Functors
using BSON
using Artifacts, LazyArtifacts
using Statistics
using MLUtils

import Functors

include("utilities.jl")

# Custom Layers
include("layers/Layers.jl")
using .Layers

# CNN models
include("convnets/alexnet.jl")
include("convnets/vgg.jl")
include("convnets/inception.jl")
include("convnets/googlenet.jl")
include("convnets/resnet.jl")
include("convnets/resnext.jl")
include("convnets/densenet.jl")
include("convnets/squeezenet.jl")
include("convnets/mobilenet.jl")
include("convnets/convnext.jl")
include("convnets/convmixer.jl")

# Other models
include("other/mlpmixer.jl")

# ViT-based models
include("vit-based/vit.jl")

include("pretrain.jl")

export AlexNet,
       VGG, VGG11, VGG13, VGG16, VGG19,
       GoogLeNet,
       ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
       Inception3, Inception4, SqueezeNet,
       ResNeXt,
       DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
       MobileNetv1, MobileNetv2, MobileNetv3,
       MLPMixer, ResMLP, gMLP,
       ViT,
       ConvMixer, ConvNeXt

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :GoogLeNet, :ResNet, :ResNeXt, :Inception3, :Inception4,
          :SqueezeNet, :DenseNet, :MobileNetv1, :MobileNetv2, :MobileNetv3,
          :MLPMixer, :ResMLP, :gMLP, :ViT, :ConvMixer, :ConvNeXt)
    @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end # module
