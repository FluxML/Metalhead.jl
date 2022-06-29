module Layers

using Flux
using CUDA
using NNlib, NNlibCUDA
using Functors
using ChainRulesCore
using Statistics
using MLUtils
using Random

include("../utilities.jl")

include("attention.jl")
include("embeddings.jl")
include("mlp-linear.jl")
include("normalise.jl")
include("conv.jl")
include("drop.jl")

export MHAttention,
       PatchEmbedding, ViPosEmbedding, ClassTokens,
       mlp_block, gated_mlp_block,
       LayerScale, DropPath,
       ChannelLayerNorm, prenorm,
       skip_identity, skip_projection,
       conv_bn, depthwise_sep_conv_bn,
       invertedresidual, squeeze_excite,
       DropBlock
end
