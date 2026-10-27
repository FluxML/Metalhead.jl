# Ported from timm.models._efficientnet_builder (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/efficientnet_* and the parameter counts of every
# registered architecture.
#
# Not ported: multi-head attention (`mha`), space-to-depth (`d`), anti-aliasing in `ir`
# and `ds` blocks.

const BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
const BN_EPS_TF_DEFAULT = 1e-3

# timm's resolve_bn_args: the BatchNorm2d arguments from `bn_momentum` and `bn_eps`.
function resolve_bn_args(bn_momentum, bn_eps)
    return merge(BATCHNORM2D,
        bn_momentum === nothing ? (;) : (; momentum = bn_momentum),
        bn_eps === nothing ? (;) : (; eps = bn_eps))
end

# A norm_layer keyword is `nothing` (BatchNorm2d), or the arguments of BatchNorm2d.
function resolve_norm_layer(norm_layer)
    norm_layer === nothing && return BATCHNORM2D
    norm_layer isa AbstractString &&
        lowercase(norm_layer) in ("batchnorm", "batchnorm2d") && return BATCHNORM2D
    norm_layer isa NamedTuple && return merge(BATCHNORM2D, norm_layer)
    return error("`norm_layer = $(repr(norm_layer))` is not supported by the Metalhead " *
                 "ports of the EfficientNet family: only BatchNorm2d is")
end

# Round the number of filters after scaling by a multiplier.
function round_channels(channels, multiplier = 1.0, divisor = 8, channel_min = nothing;
        round_limit = 0.9)
    (multiplier === nothing || iszero(multiplier)) && return channels
    return make_divisible(channels * multiplier, divisor; min_value = channel_min,
        round_limit)
end

# A kernel size, or a list of them (`3.5.7`) for a MixedConv2d.
function parse_ksize(ss)
    return all(isdigit, ss) ? parse(Int, ss) : [parse(Int, k) for k in split(ss, '.')]
end

# Decode a block definition string such as `ir_r2_k3_s2_e1_c16_se0.25_noskip` into a
# `(block args, number of repeats)` pair.
function decode_block_str(block_str::AbstractString)
    ops = split(block_str, '_')
    block_type = ops[1]
    options = Dict{String, Any}()
    skip = nothing
    for op in ops[2:end]
        if op == "noskip"
            skip = false
        elseif op == "skip"
            skip = true
        elseif startswith(op, "n")
            value = Dict("re" => relu, "r6" => relu6, "hs" => hardswish, "sw" => swish,
                "mi" => mish)
            haskey(value, op[2:end]) && (options["n"] = value[op[2:end]])
        else
            m = match(r"^(\D*)(\d.*)$", op)
            m === nothing || (options[m[1]] = m[2])
        end
    end
    act_layer = get(options, "n", nothing)
    start_kernel_size = haskey(options, "a") ? parse_ksize(options["a"]) : 1
    end_kernel_size = haskey(options, "p") ? parse_ksize(options["p"]) : 1
    force_in_chs = haskey(options, "fc") ? parse(Int, options["fc"]) : 0
    num_repeat = parse(Int, options["r"])

    block_args = Dict{Symbol, Any}(:block_type => block_type,
        :out_chs => parse(Int, options["c"]), :stride => parse(Int, options["s"]),
        :act_layer => act_layer)
    se_ratio = parse(Float64, get(options, "se", "0.0"))
    s2d = parse(Int, get(options, "d", "0")) > 0
    if block_type == "ir"
        merge!(block_args,
            Dict(:dw_kernel_size => parse_ksize(options["k"]),
                :exp_kernel_size => start_kernel_size, :pw_kernel_size => end_kernel_size,
                :exp_ratio => parse(Float64, options["e"]), :se_ratio => se_ratio,
                :noskip => skip === false, :s2d => s2d))
        haskey(options, "cc") && (block_args[:num_experts] = parse(Int, options["cc"]))
    elseif block_type == "ds" || block_type == "dsa"
        merge!(block_args,
            Dict(:dw_kernel_size => parse_ksize(options["k"]),
                :pw_kernel_size => end_kernel_size, :se_ratio => se_ratio,
                :pw_act => block_type == "dsa",
                :noskip => block_type == "dsa" || skip === false, :s2d => s2d))
    elseif block_type == "er"
        merge!(block_args,
            Dict(:exp_kernel_size => parse_ksize(options["k"]),
                :pw_kernel_size => end_kernel_size,
                :exp_ratio => parse(Float64, options["e"]),
                :force_in_chs => force_in_chs, :se_ratio => se_ratio,
                :noskip => skip === false))
    elseif block_type == "cn"
        merge!(block_args,
            Dict(:kernel_size => parse(Int, options["k"]), :skip => skip === true))
    elseif block_type == "uir"
        # `a` and `p` are the kernel sizes of the optional start and end depthwise convs
        dw_start = haskey(options, "a") ? parse_ksize(options["a"]) : 0
        dw_end = haskey(options, "p") ? parse_ksize(options["p"]) : 0
        merge!(block_args,
            Dict(:dw_kernel_size_start => dw_start,
                :dw_kernel_size_mid => parse_ksize(options["k"]),
                :dw_kernel_size_end => dw_end, :exp_ratio => parse(Float64, options["e"]),
                :se_ratio => se_ratio, :noskip => skip === false))
    elseif block_type == "mqa"
        kv_dim = parse(Int, options["d"])
        merge!(block_args,
            Dict(:dw_kernel_size => parse_ksize(options["k"]),
                :num_heads => parse(Int, options["h"]), :key_dim => kv_dim,
                :value_dim => kv_dim, :kv_stride => parse(Int, get(options, "v", "1")),
                :noskip => skip === false))
    elseif block_type == "mha"
        error("block type `mha` (multi-head attention) is not supported by the Metalhead " *
              "ports of the EfficientNet family")
    else
        error("unknown block type `$block_type`")
    end
    haskey(options, "gs") && (block_args[:group_size] = parse(Int, options["gs"]))
    return block_args, num_repeat
end

# Per-stage depth scaling: the total repeat count of the stage is scaled and then shared
# out over its block definitions, from the last one back.
function scale_stage_depth(
        stack_args, repeats, depth_multiplier = 1.0, depth_trunc = "ceil")
    num_repeat = sum(repeats)
    num_repeat_scaled = depth_trunc == "round" ?
                        max(1, round(Int, num_repeat * depth_multiplier)) :
                        ceil(Int, num_repeat * depth_multiplier)
    repeats_scaled = Int[]
    for r in reverse(repeats)
        rs = max(1, round(Int, r / num_repeat * num_repeat_scaled))
        push!(repeats_scaled, rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    end
    reverse!(repeats_scaled)
    sa_scaled = Dict{Symbol, Any}[]
    for (ba, rep) in zip(stack_args, repeats_scaled)
        append!(sa_scaled, (copy(ba) for _ in 1:rep))
    end
    return sa_scaled
end

"""
    decode_arch_def(arch_def, depth_multiplier = 1.0; depth_trunc = "ceil",
                    experts_multiplier = 1, fix_first_last = false, group_size = nothing)

Decode the block definition strings of the stages of `arch_def` (a vector of vectors of
strings) into a vector of vectors of block arguments. `depth_multiplier` is a number or one
per stage; `experts_multiplier` scales the number of CondConv experts.
"""
function decode_arch_def(arch_def, depth_multiplier = 1.0; depth_trunc = "ceil",
        experts_multiplier = 1, fix_first_last = false, group_size = nothing)
    if depth_multiplier isa Union{Tuple, AbstractVector}
        length(depth_multiplier) == length(arch_def) ||
            error("`depth_multiplier` needs one value per stage")
    else
        depth_multiplier = fill(depth_multiplier, length(arch_def))
    end
    arch_args = Vector{Dict{Symbol, Any}}[]
    for (stack_idx, (block_strings, multiplier)) in enumerate(zip(arch_def, depth_multiplier))
        stack_args = Dict{Symbol, Any}[]
        repeats = Int[]
        for block_str in block_strings
            ba, rep = decode_block_str(block_str)
            haskey(ba, :num_experts) && (ba[:num_experts] *= experts_multiplier)
            group_size === nothing || get!(ba, :group_size, group_size)
            push!(stack_args, ba)
            push!(repeats, rep)
        end
        multiplier = fix_first_last && stack_idx in (1, length(arch_def)) ? 1.0 : multiplier
        push!(arch_args, scale_stage_depth(stack_args, repeats, multiplier, depth_trunc))
    end
    return arch_args
end

"""
    EfficientNetBuilder(; output_stride = 32, pad_type = "", round_chs_fn = round_channels,
                        se_from_exp = false, act_layer = nothing, norm_layer = BATCHNORM2D,
                        aa_layer = nothing, se_layer = nothing, drop_path_rate = 0.0,
                        layer_scale_init_value = nothing)

Builds the stages of blocks from decoded block arguments. Calling it as
`builder(in_chs, block_args)` returns a vector of `Chain`s, and leaves the channel count of
the last block in `builder.in_chs`. `se_layer(channels; act_layer, rd_ratio)` creates the
squeeze-and-excitation module of a block. `aa_layer` (anti-aliasing, see `create_aa`) is
supported by the `cn`, `er`, `uir` and `mqa` blocks, and `layer_scale_init_value` applies
to the `uir` and `mqa` blocks of MobileNetV4.
"""
mutable struct EfficientNetBuilder
    output_stride::Int
    pad_type::String
    round_chs_fn::Any
    se_from_exp::Bool
    act_layer::Any
    norm_layer::Any
    aa_layer::Any
    se_layer::Any
    drop_path_rate::Float64
    layer_scale_init_value::Any
    in_chs::Int
    stage_ends::Vector{Int}
    reductions::Vector{Int}
end

function EfficientNetBuilder(; output_stride = 32, pad_type = "",
        round_chs_fn = round_channels, se_from_exp = false, act_layer = nothing,
        norm_layer = BATCHNORM2D, aa_layer = nothing, se_layer = nothing,
        drop_path_rate = 0.0, layer_scale_init_value = nothing)
    return EfficientNetBuilder(output_stride, pad_type, round_chs_fn, se_from_exp,
        act_layer, norm_layer, aa_layer, se_layer, drop_path_rate,
        layer_scale_init_value, 0, Int[], Int[])
end

const BLOCK_TYPES = Dict("ir" => InvertedResidual, "ds" => DepthwiseSeparableConv,
    "dsa" => DepthwiseSeparableConv, "er" => EdgeResidual, "cn" => ConvBnAct,
    "uir" => UniversalInvertedResidual, "mqa" => MobileAttention)

function make_block!(b::EfficientNetBuilder, ba, block_idx, block_count)
    drop_path_rate = b.drop_path_rate * block_idx / block_count
    block_type = pop!(ba, :block_type)
    in_chs = b.in_chs
    out_chs = b.round_chs_fn(pop!(ba, :out_chs))
    get(ba, :force_in_chs, 0) > 0 && (ba[:force_in_chs] = b.round_chs_fn(ba[:force_in_chs]))
    ba[:pad_type] = b.pad_type
    ba[:act_layer] = something(ba[:act_layer], b.act_layer)
    ba[:norm_layer] = b.norm_layer
    ba[:drop_path_rate] = drop_path_rate
    if b.aa_layer !== nothing
        block_type in ("cn", "er", "uir", "mqa") ||
            error("`aa_layer` is not supported in `$block_type` blocks by the Metalhead " *
                  "ports of the EfficientNet family")
        ba[:aa_layer] = b.aa_layer
    end
    if block_type in ("uir", "mqa")
        ba[:layer_scale_init_value] = b.layer_scale_init_value
        block_type == "mqa" && (ba[:use_multi_query] = true)
    end

    se_ratio = pop!(ba, :se_ratio, nothing)
    if se_ratio !== nothing && !iszero(se_ratio) && b.se_layer !== nothing
        b.se_from_exp || (se_ratio /= get(ba, :exp_ratio, 1.0))
        ba[:se_layer] = (channels; kwargs...) -> b.se_layer(
            channels; rd_ratio = se_ratio, kwargs...)
    end
    haskey(BLOCK_TYPES, block_type) || error("unknown block type `$block_type`")
    block_cls = get(ba, :num_experts, 0) > 0 ? CondConvResidual : BLOCK_TYPES[block_type]
    block = block_cls(in_chs, out_chs; ba...)
    b.in_chs = out_chs
    return block
end

function (b::EfficientNetBuilder)(in_chs, model_block_args)
    b.in_chs = in_chs
    empty!(b.stage_ends)
    empty!(b.reductions)
    total_block_count = sum(length, model_block_args)
    total_block_idx = 0
    current_stride = 2
    current_dilation = 1
    stages = []
    if model_block_args[1][1][:stride] > 1
        push!(b.stage_ends, 0)
        push!(b.reductions, current_stride)
    end
    for (stack_idx, stack_args) in enumerate(model_block_args)
        blocks = []
        for (block_idx, block_args) in enumerate(stack_args)
            ba = copy(block_args)
            ba[:stride] in (1, 2) || error("only strides of 1 and 2 are supported")
            block_idx > 1 && (ba[:stride] = 1)  # only the first block of a stage strides
            pop!(ba, :s2d, false) && error("space-to-depth (`d`) is not supported by the " *
                  "Metalhead ports of the EfficientNet family")

            next_dilation = current_dilation
            if ba[:stride] > 1
                next_output_stride = current_stride * ba[:stride]
                if next_output_stride > b.output_stride
                    next_dilation = current_dilation * ba[:stride]
                    ba[:stride] = 1
                else
                    current_stride = next_output_stride
                end
            end
            ba[:dilation] = current_dilation
            current_dilation = next_dilation

            push!(blocks, make_block!(b, ba, total_block_idx, total_block_count))
            total_block_idx += 1
        end
        push!(stages, Chain(blocks))
        if stack_idx == length(model_block_args) ||
           model_block_args[stack_idx + 1][1][:stride] > 1
            push!(b.stage_ends, stack_idx)
            push!(b.reductions, current_stride)
        end
    end
    return stages
end

block_out_chs(b::ConvBnAct) = b.bn1.chs
block_out_chs(b::Union{DepthwiseSeparableConv, EdgeResidual}) = b.bn2.chs
block_out_chs(b::Union{InvertedResidual, CondConvResidual}) = b.bn3.chs
block_out_chs(b::UniversalInvertedResidual) = b.pw_proj[:bn].chs
block_out_chs(b::MobileAttention) = size(b.attn.output[:proj].weight, 4)

function efficientnet_feature_info(m)
    return [(; num_chs = e == 0 ? m.bn1.chs : block_out_chs(m.blocks[e][end]), reduction,
                module_name = e == 0 ? "bn1" : "blocks.$(e - 1)")
            for (e, reduction) in zip(m.feature_stages.ends, m.feature_stages.reductions)]
end

function efficientnet_intermediates(m, x; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, output_fmt = "NCHW", intermediates_only::Bool = false)
    check_output_fmt(output_fmt)
    ends = m.feature_stages.ends
    take, max_index = feature_take_indices(length(ends), indices)
    stop = stop_early ? ends[max_index] : length(m.blocks)
    stem = m.bn1(m.conv_stem(x))
    outs = (stem, stage_outputs(m.blocks.layers, stem, stop)...)
    intermediates = [outs[ends[i] + 1] for i in take]
    intermediates_only && return intermediates
    return efficientnet_final_features(m, last(outs), stop), intermediates
end

# timm's efficientnet_init_weights (_init_weight_goog) for the Conv and Dense layers;
# BatchNorm already starts at weight 1 and bias 0.
function efficientnet_init_weights!(model)
    for m in Flux.modules(model)
        if m isa Conv
            fan_out = prod(size(m.weight)[1:2]) * size(m.weight, 4) ÷ m.groups
            init_parameter!(m.weight, (ds...) -> randn(Float32, ds...) .*
                                                 sqrt(2.0f0 / fan_out))
            m.bias isa AbstractArray && fill!(m.bias, 0)
        elseif m isa Dense
            init_range = 1.0f0 / sqrt(Float32(size(m.weight, 1)))
            init_parameter!(m.weight, (ds...) -> (2 .* rand(Float32, ds...) .- 1) .*
                                                 init_range)
            m.bias isa AbstractArray && fill!(m.bias, 0)
        end
    end
    # CondConv routing layers: fan-in plus fan-out (CondConv2d inits itself)
    for m in Flux.modules(model)
        m isa CondConvResidual || continue
        init_range = 1.0f0 / sqrt(Float32(sum(size(m.routing_fn.weight))))
        init_parameter!(m.routing_fn.weight,
            (ds...) -> (2 .* rand(Float32, ds...) .- 1) .* init_range)
    end
    return model
end
