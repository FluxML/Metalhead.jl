function dwsepconv_builder(block_configs, inplanes::Integer, stage_idx::Integer,
                           width_mult::Number; norm_layer = BatchNorm, kwargs...)
    block_fn, k, outplanes, stride, nrepeats, activation = block_configs[stage_idx]
    outplanes = floor(Int, outplanes * width_mult)
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][3]
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        stride = block_idx == 1 ? stride : 1
        block = Chain(block_fn((k, k), inplanes, outplanes, activation;
                               stride, pad = SamePad(), norm_layer, kwargs...)...)
        return (block,)
    end
    return get_layers, nrepeats
end

function mbconv_builder(block_configs, inplanes::Integer, stage_idx::Integer,
                        scalings::NTuple{2, Real}; norm_layer = BatchNorm,
                        divisor::Integer = 8, kwargs...)
    width_mult, depth_mult = scalings
    block_fn, k, outplanes, expansion, stride, nrepeats, reduction, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][3]
    inplanes = _round_channels(inplanes * width_mult, divisor)
    outplanes = _round_channels(outplanes * width_mult, divisor)
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, divisor)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation; norm_layer,
                         stride, reduction, no_skip = true, kwargs...)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, ceil(Int, nrepeats * depth_mult)
end

function mbconv_builder(block_configs, inplanes::Integer, stage_idx::Integer,
                        width_mult::Real; norm_layer = BatchNorm, kwargs...)
    block_fn, k, outplanes, expansion, stride, nrepeats, reduction, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][3]
    inplanes = _round_channels(inplanes * width_mult, 8)
    outplanes = _round_channels(outplanes * width_mult, 8)
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, 8)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation; norm_layer,
                         stride, reduction, no_skip = true, kwargs...)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, nrepeats
end

function fused_mbconv_builder(block_configs, inplanes::Integer,
                              stage_idx::Integer; norm_layer = BatchNorm, kwargs...)
    block_fn, k, outplanes, expansion, stride, nrepeats, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][3]
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, 8)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation;
                         norm_layer, stride, no_skip = true, kwargs...)
        return stride == 1 && inplanes == outplanes ? (identity, block) : (block,)
    end
    return get_layers, nrepeats
end

# TODO - these builders need to be more flexible to potentially specify stuff like
# activation functions and reductions that don't change
function _get_builder(::typeof(dwsep_conv_bn), block_configs, inplanes::Integer;
                      scalings::Union{Nothing, NTuple{2, Real}} = nothing,
                      width_mult::Union{Nothing, Number} = nothing, norm_layer, kwargs...)
    @assert isnothing(scalings) "dwsep_conv_bn does not support the `scalings` argument"
    return idx -> dwsepconv_builder(block_configs, inplanes, idx, width_mult; norm_layer,
                                    kwargs...)
end

function _get_builder(::typeof(mbconv), block_configs, inplanes::Integer;
                      scalings::Union{Nothing, NTuple{2, Real}} = nothing,
                      width_mult::Union{Nothing, Number} = nothing, norm_layer, kwargs...)
    if isnothing(scalings)
        return idx -> mbconv_builder(block_configs, inplanes, idx, width_mult; norm_layer,
                                     kwargs...)
    elseif isnothing(width_mult)
        return idx -> mbconv_builder(block_configs, inplanes, idx, scalings; norm_layer,
                                     kwargs...)
    else
        throw(ArgumentError("Only one of `scalings` and `width_mult` can be specified"))
    end
end

function _get_builder(::typeof(fused_mbconv), block_configs, inplanes::Integer;
                      scalings::Union{Nothing, NTuple{2, Real}} = nothing,
                      width_mult::Union{Nothing, Number} = nothing, norm_layer)
    @assert isnothing(width_mult) "fused_mbconv does not support the `width_mult` argument."
    @assert isnothing(scalings)||scalings == (1, 1) "fused_mbconv does not support the `scalings` argument"
    return idx -> fused_mbconv_builder(block_configs, inplanes, idx; norm_layer)
end

function mbconv_stack_builder(block_configs::AbstractVector{<:Tuple}, inplanes::Integer;
                              scalings::Union{Nothing, NTuple{2, Real}} = nothing,
                              width_mult::Union{Nothing, Number} = nothing,
                              norm_layer = BatchNorm, kwargs...)
    bxs = [_get_builder(block_configs[idx][1], block_configs, inplanes; scalings,
                        width_mult, norm_layer, kwargs...)(idx)
           for idx in eachindex(block_configs)]
    return (stage_idx, block_idx) -> first.(bxs)[stage_idx](block_idx), last.(bxs)
end
