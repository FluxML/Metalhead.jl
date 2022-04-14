module Layers

using Flux
using Flux: outputsize, Zygote
using Functors
using Statistics
using Random
using Distributions
using MLUtils
using NeuralAttentionlib

include("../utilities.jl")
include("attn_mask.jl")

include("attention.jl")
include("conv.jl")
include("droppath.jl")
include("embeddings.jl")
include("mlp.jl")
include("normalise.jl")
include("relative_index.jl")
include("swin_block.jl")
include("windowpartition.jl")
export MHAttention,WindowAttention
       get_attn_mask,DropPath
       PatchEmbedding,ProjectedPatchEmbedding, ViPosEmbedding, ClassTokens,PatchMerging
       mlp_block,swin_block
       ChannelLayerNorm, prenorm,
       skip_identity, skip_projection,
       conv_bn,
       invertedresidual, squeeze_excite
       window_partition,window_reverse
       get_relative_index,get_relative_bias
end
