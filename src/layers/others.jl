"""
    LayerScale(scale) 

Implements LayerScale.
([reference](https://arxiv.org/abs/2103.17239))

# Arguments
- `scale`: Scaling factor, a learnable diagonal matrix which is multiplied to the input.
"""
struct LayerScale
  scale
end

"""
    LayerScale(λ, planes::Int)

Implements LayerScale.
([reference](https://arxiv.org/abs/2103.17239))

# Arguments
- `λ`: initialisation value for the learnable diagonal matrix.
- `planes`: Size of channel dimension in the input.
"""
LayerScale(λ, planes::Int) = λ > 0 ? LayerScale(Flux.ones32(planes) * λ) : identity

@functor LayerScale
(m::LayerScale)(x) = x .* m.scale

"""
    DropPath(p)

Implements Stochastic Depth.
([reference](https://arxiv.org/abs/1603.09382))

# Arguments
- `p`: rate of Stochastic Depth.
"""
DropPath(p) = p ≥ 0 ? Dropout(p; dims = 4) : identity