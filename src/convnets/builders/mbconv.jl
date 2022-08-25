# TODO - potentially make these builders more flexible to specify stuff like
# activation functions and reductions that don't change over the stages

function dwsepconv_builder(block_configs::AbstractVector{<:Tuple}, inplanes::Integer,
                           stage_idx::Integer, scalings::NTuple{2, Real};
                           norm_layer = BatchNorm, divisor::Integer = 8, kwargs...)
    width_mult, depth_mult = scalings
    block_fn, k, outplanes, stride, nrepeats, activation = block_configs[stage_idx]
    outplanes = _round_channels(outplanes * width_mult, divisor)
    if stage_idx != 1
        inplanes = _round_channels(block_configs[stage_idx - 1][3] * width_mult, divisor)
    end
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        stride = block_idx == 1 ? stride : 1
        block = Chain(block_fn((k, k), inplanes, outplanes, activation;
                               stride, pad = SamePad(), norm_layer, kwargs...)...)
        return (block,)
    end
    return get_layers, ceil(Int, nrepeats * depth_mult)
end
_get_builder(::typeof(dwsep_conv_norm)) = dwsepconv_builder

function mbconv_builder(block_configs::AbstractVector{<:Tuple}, inplanes::Integer,
                        stage_idx::Integer, scalings::NTuple{2, Real};
                        norm_layer = BatchNorm, divisor::Integer = 8,
                        se_from_explanes::Bool = false, kwargs...)
    width_mult, depth_mult = scalings
    block_fn, k, outplanes, expansion, stride, nrepeats, reduction, activation = block_configs[stage_idx]
    # calculate number of reduced channels for squeeze-excite layer from explanes instead of inplanes
    if !isnothing(reduction)
        reduction = !se_from_explanes ? reduction * expansion : reduction
    end
    if stage_idx != 1
        inplanes = _round_channels(block_configs[stage_idx - 1][3] * width_mult, divisor)
    end
    outplanes = _round_channels(outplanes * width_mult, divisor)
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, divisor)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation; norm_layer,
                         stride, reduction, kwargs...)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, ceil(Int, nrepeats * depth_mult)
end
_get_builder(::typeof(mbconv)) = mbconv_builder

function fused_mbconv_builder(block_configs::AbstractVector{<:Tuple}, inplanes::Integer,
                              stage_idx::Integer, scalings::NTuple{2, Real};
                              norm_layer = BatchNorm, divisor::Integer = 8, kwargs...)
    width_mult, depth_mult = scalings
    block_fn, k, outplanes, expansion, stride, nrepeats, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][3]
    outplanes = _round_channels(outplanes * width_mult, divisor)
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, divisor)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation;
                         norm_layer, stride, kwargs...)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, ceil(Int, nrepeats * depth_mult)
end
_get_builder(::typeof(fused_mbconv)) = fused_mbconv_builder

function mbconv_stage_builder(block_configs::AbstractVector{<:Tuple}, inplanes::Integer,
                              scalings::NTuple{2, Real}; kwargs...)
    builders = _get_builder.(first.(block_configs))
    bxs = [builders[idx](block_configs, inplanes, idx, scalings; kwargs...)
           for idx in eachindex(block_configs)]
    return (stage_idx, block_idx) -> first.(bxs)[stage_idx](block_idx), last.(bxs)
end
