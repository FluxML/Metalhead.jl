"""
    build_invresmodel(scalings::NTuple{2, Real},
                      block_configs::AbstractVector{<:Tuple};
                      inplanes::Integer = 32, connection = +, activation = relu,
                      norm_layer = BatchNorm, divisor::Integer = 8,
                      tail_conv::Bool = true, expanded_classifier::Bool = false,
                      stochastic_depth_prob = nothing, headplanes::Integer,
                      dropout_prob = nothing, inchannels::Integer = 3,
                      nclasses::Integer = 1000, kwargs...)

Creates a generic inverted residual model structure with the specified configuration.

# Arguments

    - `scalings`: a tuple of two numbers that specify the width and depth scaling factors.
    - `block_configs`: This is a `Vector` of `Tuple`s that specifies the configuration of the
    inverted residual blocks. This can take several forms:
        + for `dwsep_conv_norm`, or depthwise separable convolutional blocks, the tuple
        should be of the form `(dwsep_conv_norm, kernel size, output channels, stride,
        number of repeats, activation function)`. For example, the following configuration
        is valid: `(dwsep_conv_norm, 3, 64, 1, 1, relu6)`.
"""
function build_invresmodel(scalings::NTuple{2, Real},
                           block_configs::AbstractVector{<:Tuple};
                           inplanes::Integer = 32, connection = +, activation = relu,
                           norm_layer = BatchNorm, divisor::Integer = 8,
                           tail_conv::Bool = true, expanded_classifier::Bool = false,
                           stochastic_depth_prob = nothing, headplanes::Integer,
                           dropout_prob = nothing, inchannels::Integer = 3,
                           nclasses::Integer = 1000, kwargs...)
    width_mult, _ = scalings
    # building first layer
    inplanes = _round_channels(inplanes * width_mult, divisor)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes, activation; stride = 2, pad = 1,
                      norm_layer))
    # building inverted residual blocks
    get_layers, block_repeats = mbconv_stage_builder(block_configs, inplanes, scalings;
                                                     stochastic_depth_prob, norm_layer,
                                                     divisor, kwargs...)
    append!(layers, cnn_stages(get_layers, block_repeats, connection))
    # building last layers
    outplanes = _round_channels(block_configs[end][3] * width_mult, divisor)
    if tail_conv
        # special case, supported fully only for MobileNetv3
        if expanded_classifier
            midplanes = _round_channels(outplanes * block_configs[end][4], divisor)
            append!(layers,
                    conv_norm((1, 1), outplanes, midplanes, activation; norm_layer))
            classifier = create_classifier(midplanes, headplanes, nclasses,
                                           (hardswish, identity); dropout_prob)
        else
            append!(layers,
                    conv_norm((1, 1), outplanes, headplanes, activation; norm_layer))
            classifier = create_classifier(headplanes, nclasses; dropout_prob)
        end
    else
        classifier = create_classifier(outplanes, nclasses; dropout_prob)
    end
    return Chain(Chain(layers...), classifier)
end
function build_invresmodel(width_mult::Real, block_configs::AbstractVector{<:Tuple};
                           kwargs...)
    return build_invresmodel((width_mult, 1), block_configs; kwargs...)
end
