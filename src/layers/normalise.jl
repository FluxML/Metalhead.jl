"""
    ChannelLayerNorm(sz::Int, λ = identity; ϵ = 1f-5)

A variant of LayerNorm where the input is normalised along the
channel dimension. The input is expected to have channel dimension with size 
`sz`. It also applies a learnable shift and rescaling.

Note that this is specifically for inputs with 4 dimensions in the format
(H, W, C, N) where H, W are the height and width of the input, C is the number
of channels, and N is the batch size.
"""
struct ChannelLayerNorm{F,D,T}
  λ::F
  diag::D
  ϵ::T
end

(m::ChannelLayerNorm)(x) = m.λ.(m.diag(MLUtils.normalise(x, dims = ndims(x) - 1, ϵ = m.ϵ)))

function ChannelLayerNorm(sz::Int, λ = identity; ϵ = 1f-5)
  diag = Flux.Diagonal(1, 1, sz)
  return ChannelLayerNorm(λ, diag, ϵ)
end

@functor ChannelLayerNorm
