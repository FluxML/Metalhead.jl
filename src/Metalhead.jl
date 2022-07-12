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
include("convnets/inception.jl")
include("convnets/googlenet.jl")
include("convnets/densenet.jl")
include("convnets/squeezenet.jl")
include("convnets/mobilenet.jl")
include("convnets/efficientnet.jl")
include("convnets/convnext.jl")
include("convnets/convmixer.jl")
## ResNets
include("convnets/resnets/core.jl")
include("convnets/resnets/resnet.jl")
include("convnets/resnets/resnext.jl")
include("convnets/resnets/seresnet.jl")
include("convnets/resnets/res2net.jl")

# Other models
include("other/mlpmixer.jl")

# ViT-based models
include("vit-based/vit.jl")

include("pretrain.jl")

export AlexNet, VGG, VGG11, VGG13, VGG16, VGG19,
       ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt, Res2Net,
       Res2NeXt,
       DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
       GoogLeNet, Inception3, Inceptionv3, Inceptionv4, InceptionResNetv2, Xception,
       SqueezeNet, MobileNetv1, MobileNetv2, MobileNetv3, EfficientNet,
       SEResNet, SEResNeXt,
       MLPMixer, ResMLP, gMLP,
       ViT,
       ConvMixer, ConvNeXt

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :ResNet, :ResNeXt, :DenseNet,
          :SEResNet, :SEResNeXt, :Res2Net, :Res2NeXt,
          :GoogLeNet, :Inceptionv3, :Inceptionv4, :InceptionResNetv2, :Xception,
          :SqueezeNet, :MobileNetv1, :MobileNetv2, :MobileNetv3, :EfficientNet,
          :MLPMixer, :ResMLP, :gMLP, :ViT, :ConvMixer, :ConvNeXt)
    @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end # module
