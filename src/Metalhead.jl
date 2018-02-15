__precompile__()

module Metalhead

using Flux, Images

export VGG19

include("utils.jl")
include("vgg19.jl")

end # module
