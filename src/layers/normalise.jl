# Utility functions for applying residual norm layers before and after a block
function residualprenorm(planes, fn; norm_layer = LayerNorm)
    return SkipConnection(Chain(norm_layer(planes), fn), +)
end
function residualpostnorm(planes, fn; norm_layer = LayerNorm)
    return SkipConnection(Chain(fn, norm_layer(planes)), +)
end

"""
    ChannelLayerNorm(sz::Integer, λ = identity; ϵ = 1.0f-6)

A variant of LayerNorm where the input is normalised along the
channel dimension. The input is expected to have channel dimension with size
`sz`. It also applies a learnable shift and rescaling after the normalization.

Note that this is specifically for inputs with 4 dimensions in the format
(H, W, C, N) where H, W are the height and width of the input, C is the number
of channels, and N is the batch size.
"""
struct ChannelLayerNorm{D, T}
    diag::D
    ϵ::T
end
@functor ChannelLayerNorm

function ChannelLayerNorm(sz::Integer, λ = identity; ϵ = 1.0f-6)
    diag = Flux.Scale(1, 1, sz, λ)
    return ChannelLayerNorm(diag, ϵ)
end

(m::ChannelLayerNorm)(x) = m.diag(Flux.normalise(x; dims = ndims(x) - 1, ϵ = m.ϵ))
