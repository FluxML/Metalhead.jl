__precompile__()
module Metalhead

using Flux, Images, BSON

# Models
export VGG19, VGG16, VGG13, VGG11, SqueezeNet, DenseNet, ResNet, GoogleNet

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
include("vgg19.jl")
include("squeezenet.jl")
include("densenet.jl")
include("resnet.jl")
include("googlenet.jl")
end # module
