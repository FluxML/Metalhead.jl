"""
    ChannelLayerNorm(sz, λ = identity; affine = true, ϵ = 1f-5)

A variant of LayerNorm where the input is normalised along the
channel dimension. The input is expected to have channel 
dimension with size `sz`.

If `affine = true` also applies a learnable shift and rescaling
as in the [`ChannelDiag`](@ref) layer.
"""
struct ChannelLayerNorm{F,D,T,N}
  λ::F
  diag::D
  ϵ::T
  size::NTuple{N,Int}
  affine::Bool
end

function(m::ChannelLayerNorm)(x)
  x = MLUtils.normalise(x, dims = ndims(x) - 1, ϵ = m.ϵ)
  m.diag === nothing ? m.λ.(x) : m.λ.(m.diag(x))
end

function ChannelLayerNorm(sz, λ = identity; affine = true, ϵ = 1f-5)
  sz = sz isa Integer ? (sz,) : sz
  diag = Flux.Diagonal(1, 1, sz...)
  return ChannelLayerNorm(λ, diag, ϵ, sz, affine)
end

@functor ChannelLayerNorm
