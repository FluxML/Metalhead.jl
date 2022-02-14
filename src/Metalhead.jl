module Metalhead

using Flux
using Flux: outputsize, Zygote
using Functors
using BSON
using Artifacts, LazyArtifacts
using Statistics
using MLUtils

import Functors

include("utilities.jl")
include("layers.jl")  

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

# Other models
include("other/mlpmixer.jl")
include("other/esrgan.jl")

# ViT-based models
include("vit-based/vit.jl")

export  AlexNet,
        VGG, VGG11, VGG13, VGG16, VGG19,
        ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        GoogLeNet, Inception3, SqueezeNet,
        DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
        ResNeXt,
        MobileNetv2, MobileNetv3,
        MLPMixer,
        ViT

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :ResNet, :GoogLeNet, :Inception3, :SqueezeNet, :DenseNet, :ResNeXt, 
          :MobileNetv2, :MobileNetv3, :MLPMixer, :ViT)
  @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end # module
