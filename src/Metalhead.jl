module Metalhead

using Flux
using Flux: outputsize, Zygote
using Functors
using BSON
using Artifacts, LazyArtifacts

import Functors

# Models
include("utilities.jl")
include("alexnet.jl")
include("vgg.jl")
include("resnet.jl")
include("googlenet.jl")
include("inception.jl")
include("squeezenet.jl")
include("densenet.jl")
include("resnext.jl")

export  AlexNet,
        VGG, VGG11, VGG13, VGG16, VGG19,
        ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        GoogLeNet, Inception3, SqueezeNet,
        DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
        ResNeXt

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :ResNet, :GoogLeNet, :Inception3, :SqueezeNet, :DenseNet, :ResNeXt)
  @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end # module
