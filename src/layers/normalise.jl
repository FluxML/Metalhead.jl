"""
    ChannelDiag(α, β)
    ChannelDiag(size::Integer)

Create an element-wise linear layer, which performs

    y = W .* x .+ b

W and b are reshaped and broadcasted versions of
α and β to match sizes along the channel dimension.

The learnable arrays are initialised `α = ones(Float32, size)` and
`β = zeros(Float32, size)`.

Used by [`ChannelLayerNorm`](@ref).
"""
struct ChannelDiag{T}
  α::T
  β::T
end

function ChannelDiag(sz)
  α = Flux.ones32(sz...)
  β = Flux.ones32(sz...)
  return ChannelDiag(α, β)
end

@functor ChannelDiag

function (a::ChannelDiag)(x)
  W = Flux.unsqueeze(Flux.unsqueeze(a.α, 1), 1)
  b = Flux.unsqueeze(Flux.unsqueeze(a.β, 1), 1)
  return W .* x .+ b
end

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
  diag = ChannelDiag(sz)
  return ChannelLayerNorm(λ, diag, ϵ, sz, affine)
end

@functor ChannelLayerNorm
