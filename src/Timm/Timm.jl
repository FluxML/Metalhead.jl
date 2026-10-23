"""
    Metalhead.Timm

Flux ports of [timm](https://github.com/huggingface/pytorch-image-models) models.
Each ported model mirrors the PyTorch module tree, so that pretrained `state_dict`s
load by name; build them with [`Metalhead.create_model`](@ref).
"""
module Timm

using Flux
using NNlib
using ConcreteStructs: @concrete
using ..Metalhead: ArchSpec, register!
import ..Metalhead: layer_params
import ..Metalhead: forward_features, forward_head

include("layers.jl")
include("resnet.jl")
include("vit.jl")
include("convnext.jl")

end # module
