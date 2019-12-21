__precompile__()
module Metalhead

using Flux, Images, ImageFiltering, BSON, REPL, Requires, Statistics
using Flux: @functor

# Models
export VGG19, SqueezeNet, DenseNet, ResNet, GoogleNet

# Useful re-export from Images
export load

# High-level classification APIs
export predict, classify

# Data Sets
export ImageNet, CIFAR10

# Data set utilities
export trainimgs, testimgs, valimgs, dataset, datasets

function __init__()
    @require TerminalExtensions="d3a6a179-465e-5219-bd3e-0137f7fd17c7" include("display/terminal_extensions.jl")
end

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
