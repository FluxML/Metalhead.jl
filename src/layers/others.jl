"""
    LayerScale(scale) 

Implements LayerScale.
([reference](https://arxiv.org/abs/2103.17239))

# Arguments
- `scale`: Scaling factor, a learnable diagonal matrix which is multiplied to the input.
"""
struct LayerScale{T<:AbstractVector{<:Real}}
    scale::T
end

"""
    LayerScale(λ, planes::Int)

Implements LayerScale.
([reference](https://arxiv.org/abs/2103.17239))

# Arguments
- `planes`: Size of channel dimension in the input.
- `λ`: initialisation value for the learnable diagonal matrix.
"""
LayerScale(planes::Int, λ) = λ > 0 ? LayerScale(Flux.ones32(planes) * λ) : identity

@functor LayerScale
(m::LayerScale)(x::AbstractArray) = m.scale .* x

"""
    DropPath(p)

Implements Stochastic Depth.
([reference](https://arxiv.org/abs/1603.09382))

# Arguments
- `p`: rate of Stochastic Depth.
"""
DropPath(p) = p ≥ 0 ? Dropout(p; dims = 4) : identity