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
using MLUtils: chunk
using Statistics: mean
using ChainRulesCore: ChainRulesCore
using ..Metalhead: ArchSpec, register!
import ..Metalhead: layer_params
import ..Metalhead: forward_features, forward_head, forward_intermediates, feature_info

include("layers.jl")
include("pos_embed.jl")
include("features.jl")
include("resnet.jl")
include("res2net.jl")
include("vit.jl")
include("convnext.jl")
include("mlp_mixer.jl")
include("convmixer.jl")
include("inception_v3.jl")
include("inception_v4.jl")
include("inception_resnet_v2.jl")
include("xception.jl")
include("vgg.jl")
include("densenet.jl")
include("efficientnet_blocks.jl")
include("efficientnet_builder.jl")
include("efficientnet.jl")
include("mobilenetv3.jl")

end # module
