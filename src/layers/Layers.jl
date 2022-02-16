module Layers

using Flux
using Flux: outputsize, Zygote
using Functors
using Statistics
using MLUtils

include("../utilities.jl")

include("attention.jl")
include("drop.jl")
include("embeddings.jl")
include("mlp.jl")
include("normalise.jl")
include("conv.jl")

export Attention, MHAttention,
       PatchEmbedding, ViPosEmbedding, ClassTokens,
       mlpblock,
       DropPath,
       ChannelLayerNorm,
       skip_identity, skip_projection,
       conv_bn
end