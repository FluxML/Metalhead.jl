__precompile__()
module Metalhead

using Flux, Images, BSON

# Models
export VGG19, squeezenet

# Useful re-export from Images
export load

# High-level classification APIs
export predict, classify

# Data Sets
export ImageNet, CIFAR10

# Data set utilities
export testimgs, valimgs, dataset, datasets

include("datasets/utils.jl")
include("model.jl")
include("utils.jl")
include("display/terminal.jl")
include("datasets/imagenet.jl")
include("datasets/cifar10.jl")
include("datasets/autodetect.jl")
include("vgg19.jl")
include("squeezenet.jl")

end # module
