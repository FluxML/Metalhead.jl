"""
    squeeze_excite(inplanes::Integer, squeeze_planes::Integer;
                   norm_layer = planes -> identity, activation = relu,
                   gate_activation = sigmoid)

    squeeze_excite(inplanes::Integer; reduction::Real = 16,
                   norm_layer = planes -> identity, activation = relu,
                   gate_activation = sigmoid)

Creates a squeeze-and-excitation layer used in MobileNets, EfficientNets and SE-ResNets.

# Arguments

  - `inplanes`: The number of input feature maps
  - `squeeze_planes`: The number of feature maps in the intermediate layers. Alternatively,
    specify the keyword arguments `reduction` and `rd_divisior`, which determine the number
    of feature maps in the intermediate layers from the number of input feature maps as:
    `squeeze_planes = _round_channels(inplanes ÷ reduction)`. (See [`_round_channels`](#) for details.)
  - `activation`: The activation function for the first convolution layer
  - `gate_activation`: The activation function for the gate layer
  - `norm_layer`: The normalization layer to be used after the convolution layers
  - `rd_planes`: The number of hidden feature maps in a squeeze and excite layer
"""
function squeeze_excite(inplanes::Integer, squeeze_planes::Integer; norm_layer = identity,
                        activation = relu, gate_activation = sigmoid)
    return SkipConnection(Chain(AdaptiveMeanPool((1, 1)),
                                conv_norm((1, 1), inplanes, squeeze_planes, activation;
                                          norm_layer)...,
                                conv_norm((1, 1), squeeze_planes, inplanes,
                                          gate_activation; norm_layer)...), .*)
end
function squeeze_excite(inplanes::Integer; reduction::Real = 16,
                        round_fn = _round_channels, kwargs...)
    return squeeze_excite(inplanes, round_fn(inplanes / reduction); kwargs...)
end

"""
    effective_squeeze_excite(inplanes, gate_activation = sigmoid)

Effective squeeze-and-excitation layer.
(reference: [CenterMask : Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667))

# Arguments

  - `inplanes`: The number of input feature maps
  - `gate_activation`: The activation function for the gate layer
"""
function effective_squeeze_excite(inplanes::Integer; gate_activation = sigmoid)
    return SkipConnection(Chain(AdaptiveMeanPool((1, 1)),
                                Conv((1, 1), inplanes => inplanes, gate_activation)), .*)
end
