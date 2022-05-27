"""
    LayerScale(λ, planes::Integer)

Creates a `Flux.Scale` layer that performs "`LayerScale`"
([reference](https://arxiv.org/abs/2103.17239)).

# Arguments
- `planes`: Size of channel dimension in the input.
- `λ`: initialisation value for the learnable diagonal matrix.
"""
LayerScale(planes::Integer, λ) =
    λ > 0 ? Flux.Scale(fill(Float32(λ), planes), false) : identity

"""
    DropPath(p)

Implements Stochastic Depth - equivalent to `Dropout(p; dims = 4)` when `p` ≥ 0.
([reference](https://arxiv.org/abs/1603.09382))

# Arguments
- `p`: rate of Stochastic Depth.
"""
DropPath(p) = p ≥ 0 ? Dropout(p; dims = 4) : identity