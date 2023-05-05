# Utility function for applying LayerNorm before a block
prenorm(planes, fn) = Chain(LayerNormV2(planes), fn)

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


"""
    LayerNormV2(size..., λ=identity; affine=true, eps=1f-5)

Same as Flux's LayerNorm but eps is added before taking the square root in the denominator.
Therefore, LayerNormV2 matches pytorch's LayerNorm.
"""
struct LayerNormV2{F,D,T,N}
  λ::F
  diag::D
  ϵ::T
  size::NTuple{N,Int}
  affine::Bool
end

function LayerNormV2(size::Tuple{Vararg{Int}}, λ=identity; affine::Bool=true, eps::Real=1f-5)
  diag = affine ? Flux.Scale(size..., λ) : λ!=identity ? Base.Fix1(broadcast, λ) : identity
  return LayerNormV2(λ, diag, eps, size, affine)
end
LayerNormV2(size::Integer...; kw...) = LayerNormV2(Int.(size); kw...)
LayerNormV2(size_act...; kw...) = LayerNormV2(Int.(size_act[1:end-1]), size_act[end]; kw...)

@functor LayerNormV2

function (a::LayerNormV2)(x::AbstractArray)
  eps = convert(float(eltype(x)), a.ϵ)  # avoids promotion for Float16 data, but should ε chage too?
  a.diag(_normalise(x; dims=1:length(a.size), eps))
end

function Base.show(io::IO, l::LayerNormV2)
  print(io, "LayerNormV2(", join(l.size, ", "))
  l.λ === identity || print(io, ", ", l.λ)
  Flux.hasaffine(l) || print(io, ", affine=false")
  print(io, ")")
end

@inline function _normalise(x::AbstractArray; dims=ndims(x), eps=Flux.ofeltype(x, 1e-5))
    μ = mean(x, dims=dims)
    σ² = var(x, dims=dims, mean=μ, corrected=false)
    return @. (x - μ) / sqrt(σ² + eps)
  end
