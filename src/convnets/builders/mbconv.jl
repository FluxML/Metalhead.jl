"""
    irblockbuilder(::typeof(irblockfn), block_configs::AbstractVector{<:Tuple},
                   inplanes::Integer, stage_idx::Integer, scalings::NTuple{2, Real};
                   stochastic_depth_prob = nothing, norm_layer = BatchNorm,
                   divisor::Integer = 8, kwargs...)

Constructs a collection of inverted residual blocks for a given stage. Note that
this function is not intended to be called directly, but rather by the [`mbconv_stage_builder`](@ref)
function. This function must only be extended if the user wishes to extend a custom inverted
residual block type.

# Arguments

  - `irblockfn`: the inverted residual block function to use in the block builder. Metalhead
    defines methods for [`dwsep_conv_norm`](@ref), [`mbconv`](@ref) and [`fused_mbconv`](@ref)
    as inverted residual blocks.
"""
function irblockbuilder(::typeof(dwsep_conv_norm), block_configs::AbstractVector{<:Tuple},
                        inplanes::Integer, stage_idx::Integer, scalings::NTuple{2, Real};
                        stochastic_depth_prob = nothing, norm_layer = BatchNorm,
                        divisor::Integer = 8, kwargs...)
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

function irblockbuilder(::typeof(mbconv), block_configs::AbstractVector{<:Tuple},
                        inplanes::Integer, stage_idx::Integer, scalings::NTuple{2, Real};
                        stochastic_depth_prob = nothing, norm_layer = BatchNorm,
                        divisor::Integer = 8, se_from_explanes::Bool = false, kwargs...)
    width_mult, depth_mult = scalings
    block_repeats = [ceil(Int, block_configs[idx][end - 2] * depth_mult)
                     for idx in eachindex(block_configs)]
    block_fn, k, outplanes, expansion, stride, _, reduction, activation = block_configs[stage_idx]
    # calculate number of reduced channels for squeeze-excite layer from explanes instead of inplanes
    if !isnothing(reduction)
        reduction = !se_from_explanes ? reduction * expansion : reduction
    end
    if stage_idx != 1
        inplanes = _round_channels(block_configs[stage_idx - 1][3] * width_mult, divisor)
    end
    outplanes = _round_channels(outplanes * width_mult, divisor)
    sdschedule = linear_scheduler(stochastic_depth_prob; depth = sum(block_repeats))
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, divisor)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation; norm_layer,
                         stride, reduction, kwargs...)
        use_skip = stride == 1 && inplanes == outplanes
        if use_skip
            schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
            drop_path = StochasticDepth(sdschedule[schedule_idx])
            return (drop_path, block)
        else
            return (block,)
        end
    end
    return get_layers, block_repeats[stage_idx]
end

function irblockbuilder(::typeof(fused_mbconv), block_configs::AbstractVector{<:Tuple},
                        inplanes::Integer, stage_idx::Integer, scalings::NTuple{2, Real};
                        stochastic_depth_prob = nothing, norm_layer = BatchNorm,
                        divisor::Integer = 8, kwargs...)
    width_mult, depth_mult = scalings
    block_repeats = [ceil(Int, block_configs[idx][end - 1] * depth_mult)
                     for idx in eachindex(block_configs)]
    block_fn, k, outplanes, expansion, stride, _, activation = block_configs[stage_idx]
    inplanes = stage_idx == 1 ? inplanes : block_configs[stage_idx - 1][3]
    outplanes = _round_channels(outplanes * width_mult, divisor)
    sdschedule = linear_scheduler(stochastic_depth_prob; depth = sum(block_repeats))
    function get_layers(block_idx::Integer)
        inplanes = block_idx == 1 ? inplanes : outplanes
        explanes = _round_channels(inplanes * expansion, divisor)
        stride = block_idx == 1 ? stride : 1
        block = block_fn((k, k), inplanes, explanes, outplanes, activation;
                         norm_layer, stride, kwargs...)
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        drop_path = StochasticDepth(sdschedule[schedule_idx])
        return stride == 1 && inplanes == outplanes ? (drop_path, block) : (block,)
    end
    return get_layers, block_repeats[stage_idx]
end

function mbconv_stage_builder(block_configs::AbstractVector{<:Tuple}, inplanes::Integer,
                              scalings::NTuple{2, Real}; kwargs...)
    bxs = [irblockbuilder(block_configs[idx][1], block_configs, inplanes, idx, scalings;
                          kwargs...) for idx in eachindex(block_configs)]
    return (stage_idx, block_idx) -> first.(bxs)[stage_idx](block_idx), last.(bxs)
end
