module Layers

using Flux
using Flux: rng_from_array
using CUDA
using NNlib, NNlibCUDA
using Functors
using ChainRulesCore
using Statistics
using MLUtils
using PartialFunctions
using Random

include("../utilities.jl")

include("attention.jl")
export MHAttention

include("conv.jl")
export conv_norm, depthwise_sep_conv_norm, invertedresidual

include("drop.jl")
export DropBlock, DropPath

include("embeddings.jl")
export PatchEmbedding, ViPosEmbedding, ClassTokens

include("mlp.jl")
export mlp_block, gated_mlp_block, create_fc, create_classifier

include("normalise.jl")
export prenorm, ChannelLayerNorm

include("pool.jl")
export AdaptiveMeanMaxPool

include("scale.jl")
export LayerScale, inputscale

include("selayers.jl")
export squeeze_excite, effective_squeeze_excite

end
