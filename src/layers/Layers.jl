module Layers

using Flux
using Flux: default_rng_value
using NNlib
using Functors
using ChainRulesCore
using Statistics
using MLUtils
using PartialFunctions
using Random

import Flux.testmode!

include("utilities.jl")

include("attention.jl")
export MultiHeadSelfAttention

include("conv.jl")
export conv_norm, basic_conv_bn, dwsep_conv_norm

include("drop.jl")
export DropBlock, StochasticDepth

include("embeddings.jl")
export PatchEmbedding, ViPosEmbedding, ClassTokens

include("mbconv.jl")
export mbconv, fused_mbconv

include("mlp.jl")
export mlp_block, gated_mlp_block

include("classifier.jl")
export create_classifier

include("normalise.jl")
export prenorm, ChannelLayerNorm

include("pool.jl")
export AdaptiveMeanMaxPool

include("scale.jl")
export LayerScale, inputscale

include("selayers.jl")
export squeeze_excite, effective_squeeze_excite

end
