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

include("embeddings.jl")
export PatchEmbedding, PositionalEmbedding, ClassTokens

include("mlp.jl")
export mlp_block, gated_mlp_block, create_fc, create_classifier

include("normalise.jl")
export residualprenorm, residualpostnorm, ChannelLayerNorm

include("conv.jl")
export conv_norm, depthwise_sep_conv_bn, invertedresidual, skip_identity, skip_projection

include("drop.jl")
export DropBlock, DropPath, droppath_rates

include("selayers.jl")
export squeeze_excite, effective_squeeze_excite

include("scale.jl")
export LayerScale, inputscale

include("pool.jl")
export AdaptiveMeanMaxPool

end
