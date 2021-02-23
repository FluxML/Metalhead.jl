module Metalhead

using Flux
using Flux: outputsize

# Models
include("utilities.jl")
include("alexnet.jl")
include("vgg.jl")
include("resnet.jl")
include("googlenet.jl")
include("inception.jl")
include("squeezenet.jl")
include("densenet.jl")

export  alexnet,
        vgg11, vgg11bn, vgg13, vgg13bn, vgg16, vgg16bn, vgg19, vgg19bn,
        resnet18, resnet34, resnet50, resnet101, resnet152,
        googlenet, inception3, squeezenet,
        densenet121, densenet161, densenet169, densenet201

end # module
