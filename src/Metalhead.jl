__precompile__()

module Metalhead

using Flux, Images, BSON

export load, VGG19

const imagenet_classes = split(String(read(joinpath(@__DIR__, "..", "imagenet_classes.txt"))),
                               "\n", keep = false)

include("utils.jl")
include("vgg19.jl")

end # module
