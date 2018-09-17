__precompile__()
module Metalhead

using Flux, Images, BSON
using Flux: @treelike

# Models
export VGG19, VGG19_BN, VGG16, VGG16_BN, VGG13, VGG13_BN, VGG11, VGG11_BN,
       SqueezeNet, DenseNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
       GoogleNet

# Trained Models Loader
export trained

# Useful re-export from Images
export load

# High-level classification APIs
export predict, classify

# Data Sets
export ImageNet, CIFAR10

# Data set utilities
export trainimgs, testimgs, valimgs, dataset, datasets

include("datasets/utils.jl")
include("model.jl")
include("utils.jl")
include("display/terminal.jl")
include("datasets/imagenet.jl")
include("datasets/cifar10.jl")
include("datasets/autodetect.jl")
include("vgg.jl")
include("squeezenet.jl")
include("densenet.jl")
include("resnet.jl")
include("googlenet.jl")
end # module
