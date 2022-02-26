"""
		mlp_block(inplanes, hidden_planes; outplanes = inplanes, dropout = 0., activation = gelu)

Feedforward block used in many MLPMixer-like and vision-transformer models.

# Arguments
- `inplanes`: Number of dimensions in the input.
- `hidden_planes`: Number of dimensions in the intermediate layer.
- `outplanes`: Number of dimensions in the output - by default it is the same as `inplanes`.
- `dropout`: Dropout rate.
- `activation`: Activation function to use.
"""
function mlp_block(inplanes, hidden_planes; outplanes = inplanes, dropout = 0., activation = gelu)
  Chain(Dense(inplanes, hidden_planes, activation), Dropout(dropout),
        Dense(hidden_planes, outplanes), Dropout(dropout))
end

"""
    gated_mlp(inplanes, hidden_planes; outplanes = inplanes, dropout = 0., activation = gelu, 
							gate_layer = identity)

Feedforward block based on the implementation in the paper "Pay Attention to MLPs".
([reference](https://arxiv.org/abs/2105.08050))

# Arguments
- `inplanes`: Number of dimensions in the input.
- `hidden_planes`: Number of dimensions in the intermediate layer.
- `outplanes`: Number of dimensions in the output - by default it is the same as `inplanes`.
- `dropout`: Dropout rate.
- `activation`: Activation function to use.
- `gate_layer`: Layer to use for the gating.
"""
function gated_mlp(inplanes, hidden_planes; outplanes = inplanes, dropout = 0., activation = gelu, 
									 gate_layer = identity)
	layers = []
	push!(layers, Dense(inplanes, hidden_planes, activation))
	push!(layers, Dropout(dropout))
	if typeof(gate_layer) != typeof(identity)
		@assert hidden_planes % 2 == 0 "`hidden_planes` must be even for gated MLP"
		gate_layer = gate_layer(hidden_planes)
		hidden_planes ÷= 2
	end
	push!(layers, gate_layer)
	push!(layers, Dense(hidden_planes, outplanes))
	push!(layers, Dropout(dropout))
	return Chain(layers...)
end
