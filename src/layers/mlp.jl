"""
    mlp_block(planes, hidden_planes; dropout = 0., dense = Dense, activation = gelu)

Feedforward block used in many vision transformer-like models.

# Arguments
- `planes`: Number of dimensions in the input and output.
- `hidden_planes`: Number of dimensions in the intermediate layer.
- `dropout`: Dropout rate.
- `dense`: Type of dense layer to use in the feedforward block.
- `activation`: Activation function to use.
"""
function mlp_block(planes, hidden_planes; dropout = 0., dense = Dense, activation = gelu)
  Chain(dense(planes, hidden_planes, activation), Dropout(dropout),
        dense(hidden_planes, planes), Dropout(dropout))
end

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
LayerScale(λ, planes::Int) = LayerScale(Flux.ones32(planes) * λ)

@functor LayerScale
(m::LayerScale)(x) = x .* m.scale

"""
    DropPath(p)

Implements Stochastic Depth.
([reference](https://arxiv.org/abs/1603.09382))

# Arguments
- `p`: rate of Stochastic Depth.
"""
DropPath(p) = Dropout(p; dims = 4)