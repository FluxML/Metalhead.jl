function mbconv_builder(block_configs::AbstractVector{<:Tuple},
                        inplanes::Integer, stage_idx::Integer;
                        scalings::NTuple{2, Real} = (1, 1), norm_layer = BatchNorm,
                        round_fn = planes -> _round_channels(planes, 8))
    width_mult, depth_mult = scalings
    k, outplanes, expansion, stride, nrepeats, reduction, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][2]
    inplanes = round_fn(inplanes * width_mult)
    outplanes = _round_channels(outplanes * width_mult, 8)
    function get_layers(block_idx)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, 8)
        stride = block_idx == 1 ? stride : 1
        block = mbconv((k, k), inplanes, explanes, outplanes, activation; norm_layer,
                       stride, reduction)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, ceil(Int, nrepeats * depth_mult)
end

function fused_mbconv_builder(block_configs::AbstractVector{<:Tuple},
                              inplanes::Integer, stage_idx::Integer;
                              scalings::NTuple{2, Real} = (1, 1), norm_layer = BatchNorm)
    k, outplanes, expansion, stride, nrepeats, _, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][2]
    function get_layers(block_idx)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, 8)
        stride = block_idx == 1 ? stride : 1
        block = fused_mbconv((k, k), inplanes, explanes, outplanes, activation;
                             norm_layer, stride)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, nrepeats
end

function mbconv_stack_builder(block_configs::AbstractVector{<:Tuple},
                              residual_fns::AbstractVector; inplanes::Integer,
                              scalings::NTuple{2, Real} = (1, 1),
                              norm_layer = BatchNorm)
    bxs = [residual_fn(block_configs, inplanes, stage_idx; scalings, norm_layer)
           for (stage_idx, residual_fn) in enumerate(residual_fns)]
    return (stage_idx, block_idx) -> first.(bxs)[stage_idx](block_idx), last.(bxs)
end

function efficientnet(block_configs::AbstractVector{<:Tuple},
                      residual_fns::AbstractVector; inplanes::Integer,
                      scalings::NTuple{2, Real} = (1, 1),
                      headplanes::Integer = block_configs[end][3] * 4,
                      norm_layer = BatchNorm, dropout_rate = nothing,
                      inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = []
    # stem of the model
    append!(layers,
            conv_norm((3, 3), inchannels, _round_channels(inplanes * scalings[1], 8),
                      swish; norm_layer, stride = 2, pad = SamePad()))
    # building inverted residual blocks
    get_layers, block_repeats = mbconv_stack_builder(block_configs, residual_fns;
                                                     inplanes, scalings, norm_layer)
    append!(layers, resnet_stages(get_layers, block_repeats, +))
    # building last layers
    append!(layers,
            conv_norm((1, 1), _round_channels(block_configs[end][2] * scalings[1], 8),
                      headplanes, swish; pad = SamePad()))
    return Chain(Chain(layers...), create_classifier(headplanes, nclasses; dropout_rate))
end
