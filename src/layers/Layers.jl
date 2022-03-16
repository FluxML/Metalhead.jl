module Layers

using Flux
using Flux: outputsize, Zygote
using Functors
using Statistics
using MLUtils
using NeuralAttentionlib

include("../utilities.jl")

include("attention.jl")
include("embeddings.jl")
include("mlp.jl")
include("normalise.jl")
include("conv.jl")
include("others.jl")

export MHAttention,
       PatchEmbedding, ViPosEmbedding, ClassTokens,
       mlp_block, gated_mlp_block,
       LayerScale, DropPath,
       ChannelLayerNorm, prenorm,
       skip_identity, skip_projection,
       conv_bn,
       invertedresidual, squeeze_excite
end
