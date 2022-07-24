"""
    inputscale(λ; activation = identity)

Scale the input by a scalar `λ` and applies an activation function to it.
Equivalent to `activation.(λ .* x)`.
"""
inputscale(λ; activation = identity) = _input_scale$(λ, activation)
_input_scale(λ, activation, x) = activation.(λ .* x)
_input_scale(λ, ::typeof(identity), x) = λ .* x

"""
    LayerScale(λ, planes::Integer)

Creates a `Flux.Scale` layer that performs "`LayerScale`"
([reference](https://arxiv.org/abs/2103.17239)).

# Arguments

  - `planes`: Size of channel dimension in the input.
  - `λ`: initialisation value for the learnable diagonal matrix.
"""
function LayerScale(planes::Integer, λ = 1.0f-5)
    return λ > 0 ? Flux.Scale(fill(Float32(λ), planes), false) : identity
end
