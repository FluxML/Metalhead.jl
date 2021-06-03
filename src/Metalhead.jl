module Metalhead

using Flux
using Flux: outputsize, Zygote
using BSON
using Pkg.Artifacts

# Models
include("utilities.jl")
include("alexnet.jl")
include("vgg.jl")
include("resnet.jl")
include("googlenet.jl")
include("inception.jl")
include("squeezenet.jl")
include("densenet.jl")

export  AlexNet,
        VGG, VGG11, VGG13, VGG16, VGG19,
        ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        GoogLeNet, Inception3, SqueezeNet,
        DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201

end # module
