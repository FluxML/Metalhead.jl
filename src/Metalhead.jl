module Metalhead

using Flux
using Flux: Zygote, outputsize
using Functors
using BSON
using Artifacts, LazyArtifacts
using Statistics
using MLUtils
using PartialFunctions
using Random

import Functors

include("utilities.jl")

# Custom Layers
include("layers/Layers.jl")
using .Layers

# CNN models
include("convnets/alexnet.jl")
include("convnets/vgg.jl")
## ResNets
include("convnets/resnets/core.jl")
include("convnets/resnets/resnet.jl")
include("convnets/resnets/resnext.jl")
include("convnets/resnets/seresnet.jl")
## Inceptions
include("convnets/inception/googlenet.jl")
include("convnets/inception/inceptionv3.jl")
include("convnets/inception/inceptionv4.jl")
include("convnets/inception/inceptionresnetv2.jl")
include("convnets/inception/xception.jl")
## MobileNets
include("convnets/mobilenet/mobilenetv1.jl")
include("convnets/mobilenet/mobilenetv2.jl")
include("convnets/mobilenet/mobilenetv3.jl")
## Others
include("convnets/densenet.jl")
include("convnets/squeezenet.jl")
include("convnets/efficientnet.jl")
include("convnets/convnext.jl")
include("convnets/convmixer.jl")

# Mixers
include("mixers/core.jl")
include("mixers/mlpmixer.jl")
include("mixers/resmlp.jl")
include("mixers/gmlp.jl")

# ViTs
include("vit-based/vit.jl")

# Load pretrained weights
include("pretrain.jl")

export AlexNet, VGG, VGG11, VGG13, VGG16, VGG19,
       ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt,
       DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
       GoogLeNet, Inception3, Inceptionv3, Inceptionv4, InceptionResNetv2, Xception,
       SqueezeNet, MobileNetv1, MobileNetv2, MobileNetv3, EfficientNet,
       WideResNet, SEResNet, SEResNeXt,
       MLPMixer, ResMLP, gMLP,
       ViT,
       ConvMixer, ConvNeXt

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :ResNet, :ResNeXt, :DenseNet, :SEResNet, :SEResNeXt,
          :GoogLeNet, :Inceptionv3, :Inceptionv4, :InceptionResNetv2, :Xception,
          :SqueezeNet, :MobileNetv1, :MobileNetv2, :MobileNetv3, :EfficientNet,
          :MLPMixer, :ResMLP, :gMLP, :ViT, :ConvMixer, :ConvNeXt)
    @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end # module
