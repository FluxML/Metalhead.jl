"""
    mlp_block(inplanes::Integer, hidden_planes::Integer, outplanes::Integer = inplanes; 
              dropout = 0., activation = gelu)

Feedforward block used in many MLPMixer-like and vision-transformer models.

# Arguments

  - `inplanes`: Number of dimensions in the input.
  - `hidden_planes`: Number of dimensions in the intermediate layer.
  - `outplanes`: Number of dimensions in the output - by default it is the same as `inplanes`.
  - `dropout`: Dropout rate.
  - `activation`: Activation function to use.
"""
function mlp_block(inplanes::Integer, hidden_planes::Integer, outplanes::Integer = inplanes;
                   dropout = 0.0, activation = gelu)
    return Chain(Dense(inplanes, hidden_planes, activation), Dropout(dropout),
                 Dense(hidden_planes, outplanes), Dropout(dropout))
end

"""
    gated_mlp(gate_layer, inplanes::Integer, hidden_planes::Integer, 
              outplanes::Integer = inplanes; dropout = 0., activation = gelu)

Feedforward block based on the implementation in the paper "Pay Attention to MLPs".
([reference](https://arxiv.org/abs/2105.08050))

# Arguments

  - `gate_layer`: Layer to use for the gating.
  - `inplanes`: Number of dimensions in the input.
  - `hidden_planes`: Number of dimensions in the intermediate layer.
  - `outplanes`: Number of dimensions in the output - by default it is the same as `inplanes`.
  - `dropout`: Dropout rate.
  - `activation`: Activation function to use.
"""
function gated_mlp_block(gate_layer, inplanes::Integer, hidden_planes::Integer,
                         outplanes::Integer = inplanes; dropout = 0.0, activation = gelu)
    @assert hidden_planes % 2==0 "`hidden_planes` must be even for gated MLP"
    return Chain(Dense(inplanes, hidden_planes, activation),
                 Dropout(dropout),
                 gate_layer(hidden_planes),
                 Dense(hidden_planes ÷ 2, outplanes),
                 Dropout(dropout))
end
gated_mlp_block(::typeof(identity), args...; kwargs...) = mlp_block(args...; kwargs...)
