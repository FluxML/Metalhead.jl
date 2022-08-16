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
