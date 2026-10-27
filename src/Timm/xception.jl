# Ported from timm.models.xception_aligned (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/xception_* and test/references/timm--xception*.

# Not named SeparableConv2d, which is also a timm.layers class with other keys.
@concrete struct XceptionSeparableConv2d
    conv_dw
    bn_dw
    act_dw
    conv_pw
    bn_pw
    act_pw
end
Flux.@layer XceptionSeparableConv2d

function XceptionSeparableConv2d(in_chs, out_chs; kernel_size = 3, stride = 1,
        dilation = 1, padding = "", act_layer = identity, eps = 1e-5)
    bn(c) = init_layer(BatchNorm, c; eps = Float32(eps))
    conv_dw = create_conv2d(
        in_chs, in_chs, kernel_size; stride, padding, dilation, depthwise = true)
    return XceptionSeparableConv2d(conv_dw, bn(in_chs), act_layer,
        create_conv2d(in_chs, out_chs, 1), bn(out_chs), act_layer)
end

function (m::XceptionSeparableConv2d)(x)
    x = m.act_dw(m.bn_dw(m.conv_dw(x)))
    return m.act_pw(m.bn_pw(m.conv_pw(x)))
end

@concrete struct XceptionPreSeparableConv2d
    norm
    conv_dw
    conv_pw
end
Flux.@layer XceptionPreSeparableConv2d

function XceptionPreSeparableConv2d(in_chs, out_chs; kernel_size = 3, stride = 1,
        dilation = 1, padding = "", first_act = true, eps = 1e-5)
    conv_dw = create_conv2d(
        in_chs, in_chs, kernel_size; stride, padding, dilation, depthwise = true)
    return XceptionPreSeparableConv2d(
        first_act ? init_layer(BatchNorm, in_chs, relu; eps = Float32(eps)) : identity, conv_dw,
        create_conv2d(in_chs, out_chs, 1))
end

(m::XceptionPreSeparableConv2d)(x) = m.conv_pw(m.conv_dw(m.norm(x)))

@concrete struct XceptionModule
    shortcut
    stack
    drop_path
    no_skip
end
Flux.@layer XceptionModule

as_triple(x::Integer) = (x, x, x)
as_triple(x) = Tuple(x)

function XceptionModule(in_chs, out_chs; stride = 1, dilation = 1, pad_type = "",
        start_with_relu = true, no_skip = false, eps = 1e-5, drop_path = nothing)
    out_chs = as_triple(out_chs)
    shortcut = nothing
    if !no_skip && (out_chs[end] != in_chs || stride != 1)
        shortcut = conv_norm_act(in_chs, out_chs[end], 1; stride, apply_act = false, eps)
    end
    layers = Pair{Symbol, Any}[]
    for i in 1:3
        start_with_relu && push!(layers, Symbol(:act, i) => relu)
        push!(layers,
            Symbol(:conv, i) => XceptionSeparableConv2d(in_chs, out_chs[i];
                stride = i == 3 ? stride : 1, dilation, padding = pad_type,
                act_layer = start_with_relu ? identity : relu, eps))
        in_chs = out_chs[i]
    end
    return XceptionModule(shortcut, Chain((; layers...)), drop_path, no_skip)
end

function (m::XceptionModule)(x)
    skip = x
    x = m.stack(x)
    m.shortcut === nothing || (skip = m.shortcut(skip))
    if !m.no_skip
        m.drop_path === nothing || (x = m.drop_path(x))
        x = x .+ skip
    end
    return x
end

@concrete struct PreXceptionModule
    shortcut
    norm
    stack
    drop_path
    no_skip
end
Flux.@layer PreXceptionModule

function PreXceptionModule(in_chs, out_chs; stride = 1, dilation = 1, pad_type = "",
        no_skip = false, eps = 1e-5, drop_path = nothing)
    out_chs = as_triple(out_chs)
    shortcut = !no_skip && (out_chs[end] != in_chs || stride != 1) ?
               create_conv2d(in_chs, out_chs[end], 1; stride) : identity
    norm = init_layer(BatchNorm, in_chs, relu; eps = Float32(eps))
    stack = Chain(
        (;
        (Symbol(:conv, i) => XceptionPreSeparableConv2d(
             i == 1 ? in_chs : out_chs[i - 1], out_chs[i]; stride = i == 3 ? stride : 1,
             dilation, padding = pad_type, first_act = i > 1, eps) for i in 1:3)...))
    return PreXceptionModule(shortcut, norm, stack, drop_path, no_skip)
end

function (m::PreXceptionModule)(x)
    x = m.norm(x)
    skip = x
    x = m.stack(x)
    if !m.no_skip
        m.drop_path === nothing || (x = m.drop_path(x))
        x = x .+ m.shortcut(skip)
    end
    return x
end

@concrete struct XceptionAligned
    stem
    blocks
    act
    head
    feature_stages
end
Flux.@layer XceptionAligned

"""
    xception_aligned(; block_cfg, num_classes = 1000, in_chans = 3, kwargs...)

Port of timm's `XceptionAligned` (Aligned Xception, with the pre-activation variant).
Keyword arguments are those of the timm class, except that `norm_eps` stands for the `eps`
of a `norm_layer = partial(nn.BatchNorm2d, eps = ...)`. `block_cfg` is a sequence of named
tuples `(in_chs, out_chs, stride)` with the optional `no_skip` and `start_with_relu`.
"""
function xception_aligned(;
        block_cfg, num_classes = 1000, in_chans = 3, output_stride = 32,
        preact = false, act_layer = "relu", norm_layer = "batchnorm2d", norm_eps = 1e-5,
        drop_rate = 0.0, drop_path_rate = 0.0, global_pool = "avg")
    check_unsupported("XceptionAligned"; act_layer = (act_layer, "relu"),
        norm_layer = (norm_layer, "batchnorm2d"), global_pool = (global_pool, "avg"))
    output_stride in (8, 16, 32) || error("`output_stride` must be 8, 16 or 32")
    eps = norm_eps
    stem = Chain(conv_norm_act(in_chans, 32, 3; stride = 2, eps),
        preact ? create_conv2d(32, 64, 3) : conv_norm_act(32, 64, 3; eps))
    curr_dilation, curr_stride = 1, 2
    stage_ends, reductions = Int[], Int[]
    num_blocks = length(block_cfg)
    blocks = map(enumerate(block_cfg)) do (i, b)
        dilation, stride = curr_dilation, b.stride
        if stride > 1
            push!(stage_ends, i)
            push!(reductions, curr_stride)
            next_stride = curr_stride * stride
            if next_stride > output_stride
                curr_dilation *= stride
                stride = 1
            else
                curr_stride = next_stride
            end
        end
        args = (; stride, dilation, no_skip = get(b, :no_skip, false), eps,
            drop_path = drop_path(drop_path_rate * (i - 1) / (num_blocks - 1)))
        return preact ? PreXceptionModule(b.in_chs, b.out_chs; args...) :
               XceptionModule(b.in_chs, b.out_chs;
            start_with_relu = get(b, :start_with_relu, true), args...)
    end
    num_features = as_triple(block_cfg[end].out_chs)[end]
    push!(stage_ends, num_blocks)
    push!(reductions, curr_stride)
    return XceptionAligned(stem, Chain(blocks), preact ? relu : identity,
        ClassifierHead(num_features, num_classes; drop_rate),
        (; ends = Tuple(stage_ends), reductions = Tuple(reductions)))
end

forward_features(m::XceptionAligned, x) = m.act(m.blocks(m.stem(x)))
forward_head(m::XceptionAligned, x; pre_logits::Bool = false) = m.head(x; pre_logits)
(m::XceptionAligned)(x) = forward_head(m, forward_features(m, x))

# A strided block's tap precedes conv3; its shortcut still takes the original block input.
function xception_block_intermediate(m::XceptionModule, x; intermediate_only = false)
    tap = m.stack[:act3](m.stack[:conv2](m.stack[:act2](m.stack[:conv1](m.stack[:act1](x)))))
    intermediate_only && return tap, tap
    y = m.stack[:conv3](tap)
    if !m.no_skip
        skip = m.shortcut === nothing ? x : m.shortcut(x)
        y = (m.drop_path === nothing ? y : m.drop_path(y)) .+ skip
    end
    return y, tap
end

function xception_block_intermediate(m::PreXceptionModule, x; intermediate_only = false)
    x = m.norm(x)
    tap = m.stack[:conv2](m.stack[:conv1](x))
    intermediate_only && return tap, tap
    y = m.stack[:conv3](tap)
    if !m.no_skip
        y = (m.drop_path === nothing ? y : m.drop_path(y)) .+ m.shortcut(x)
    end
    return y, tap
end

function forward_intermediates(
        m::XceptionAligned, x; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, output_fmt = "NCHW", intermediates_only::Bool = false)
    check_output_fmt(output_fmt)
    ends = m.feature_stages.ends
    take, max_index = feature_take_indices(length(ends), indices)
    stop = stop_early ? ends[max_index] : length(m.blocks)
    x = m.stem(x)
    intermediates = ()
    for i in 1:stop
        tap_index = findfirst(==(i), ends[1:(end - 1)])
        if tap_index !== nothing && tap_index in take
            early = stop_early && tap_index == max_index
            x, tap = xception_block_intermediate(m.blocks[i], x; intermediate_only = early)
            intermediates = (intermediates..., tap)
        else
            x = m.blocks[i](x)
        end
        if i == length(m.blocks) && length(ends) in take
            intermediates = (intermediates..., x)
        end
    end
    intermediates_only && return collect(intermediates)
    complete = !stop_early || max_index == length(ends)
    return complete ? m.act(x) : x, collect(intermediates)
end

function feature_info(m::XceptionAligned)
    ends = m.feature_stages.ends
    return map(enumerate(zip(ends, m.feature_stages.reductions))) do (i, (e, reduction))
        block = m.blocks[e]
        final = i == length(ends)
        conv = block.stack[final ? :conv3 : :conv2]
        chs = size(conv.conv_pw.weight, 4)
        leaf = block isa PreXceptionModule ? "stack.conv2" : "stack.act3"
        return (; num_chs = chs, reduction,
            module_name = final ? "blocks.$(e - 1)" : "blocks.$(e - 1).$leaf")
    end
end

const XCEPTION_REFERENCES = Dict(
    "xception41" =>
        ["timm/xception41.tf_in1k" => "8a17189361e63c972815ef62f2a30dd5b9f393b1"],
    "xception41p" =>
        ["timm/xception41p.ra3_in1k" => "39c1434e60cd96017b21dd204b26df2716f24136"]
)

let entry = ((in_chs = 64, out_chs = 128, stride = 2),
        (in_chs = 128, out_chs = 256, stride = 2),
        (in_chs = 256, out_chs = 728, stride = 2)),
    entry71 = ((in_chs = 64, out_chs = 128, stride = 2),
        (in_chs = 128, out_chs = 256, stride = 1),
        (in_chs = 256, out_chs = 256, stride = 2),
        (in_chs = 256, out_chs = 728, stride = 1),
        (in_chs = 728, out_chs = 728, stride = 2)),
    middle(n) = ntuple(_ -> (in_chs = 728, out_chs = 728, stride = 1), n),
    exit_flow = ((in_chs = 728, out_chs = (728, 1024, 1024), stride = 2),
        (in_chs = 1024, out_chs = (1536, 1536, 2048), stride = 1,
            no_skip = true, start_with_relu = false)),
    exit_flow_p = ((in_chs = 728, out_chs = (728, 1024, 1024), stride = 2),
        (in_chs = 1024, out_chs = (1536, 1536, 2048), stride = 1, no_skip = true))

    for (name, args) in [
        "xception41" =>
            (; block_cfg = (entry..., middle(8)..., exit_flow...), norm_eps = 1e-3),
        "xception65" =>
            (; block_cfg = (entry..., middle(16)..., exit_flow...), norm_eps = 1e-3),
        "xception71" =>
            (; block_cfg = (entry71..., middle(16)..., exit_flow...), norm_eps = 1e-3),
        "xception41p" =>
            (; block_cfg = (entry..., middle(8)..., exit_flow_p...), preact = true),
        "xception65p" => (; block_cfg = (entry..., middle(16)..., exit_flow_p...),
            preact = true, norm_eps = 1e-3)
    ]
        register!(ArchSpec(TIMM, name, xception_aligned; family = "xception_aligned",
            defaults = args,
            references = get(XCEPTION_REFERENCES, name, [])))
    end
end
