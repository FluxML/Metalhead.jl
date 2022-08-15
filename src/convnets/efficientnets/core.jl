function mbconv_builder(block_configs::AbstractVector{NTuple{6, Int}},
                        stage_idx::Integer; scalings::NTuple{2, Real} = (1, 1),
                        norm_layer = BatchNorm)
    depth_mult, width_mult = scalings
    k, inplanes, outplanes, expansion, stride, nrepeats = block_configs[stage_idx]
    inplanes = _round_channels(inplanes * width_mult, 8)
    outplanes = _round_channels(outplanes * width_mult, 8)
    function get_layers(block_idx)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, 8)
        stride = block_idx == 1 ? stride : 1
        block = mbconv((k, k), inplanes, explanes, outplanes, swish; norm_layer,
                       stride, reduction = 4)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, ceil(Int, nrepeats * depth_mult)
end

function fused_mbconv_builder(block_configs::AbstractVector{NTuple{6, Int}},
                              stage_idx::Integer; norm_layer = BatchNorm)
    k, inplanes, outplanes, expansion, stride, nrepeats = block_configs[stage_idx]
    function get_layers(block_idx)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, 8)
        stride = block_idx == 1 ? stride : 1
        block = fused_mbconv((k, k), inplanes, explanes, outplanes, swish;
                             norm_layer, stride)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, nrepeats
end

function efficientnet_builder(block_configs::AbstractVector{NTuple{6, Int}},
                              residual_fns::AbstractVector;
                              scalings::NTuple{2, Real} = (1, 1), norm_layer = BatchNorm)
    bxs = [residual_fn(block_configs, stage_idx; scalings, norm_layer)
           for (stage_idx, residual_fn) in enumerate(residual_fns)]
    return (stage_idx, block_idx) -> first.(bxs)[stage_idx](block_idx), last.(bxs)
end

function efficientnet(block_configs::AbstractVector{NTuple{6, Int}},
                      residual_fns::AbstractVector; scalings::NTuple{2, Real} = (1, 1),
                      headplanes::Integer = _round_channels(block_configs[end][3] *
                                                            scalings[2], 8) * 4,
                      norm_layer = BatchNorm, dropout_rate = nothing,
                      inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = []
    # stem of the model
    append!(layers,
            conv_norm((3, 3), inchannels, block_configs[1][2], swish; norm_layer,
                      stride = 2, pad = SamePad()))
    # building inverted residual blocks
    get_layers, block_repeats = efficientnet_builder(block_configs, residual_fns;
                                                     scalings, norm_layer)
    append!(layers, resnet_stages(get_layers, block_repeats, +))
    # building last layers
    append!(layers,
            conv_norm((1, 1), _round_channels(block_configs[end][3] * scalings[2], 8),
                      headplanes, swish; pad = SamePad()))
    return Chain(Chain(layers...), create_classifier(headplanes, nclasses; dropout_rate))
end
