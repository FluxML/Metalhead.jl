module Layers

using Flux
using Flux: rng_from_array
using CUDA
using NNlib, NNlibCUDA
using Functors
using ChainRulesCore
using Statistics
using MLUtils
using Random

include("../utilities.jl")

include("attention.jl")
export MHAttention

include("embeddings.jl")
export PatchEmbedding, ViPosEmbedding, ClassTokens

include("mlp-linear.jl")
export mlp_block, gated_mlp_block, LayerScale

include("normalise.jl")
export prenorm, ChannelLayerNorm

include("conv.jl")
export conv_bn, depthwise_sep_conv_bn, invertedresidual
skip_identity, skip_projection

include("drop.jl")
export DropPath, DropBlock

include("selayers.jl")
export squeeze_excite, effective_squeeze_excite

include("classifier.jl")
export create_classifier

include("pool.jl")
export AdaptiveMeanMaxPool, AdaptiveCatMeanMaxPool
SelectAdaptivePool

end
