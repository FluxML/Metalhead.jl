# Ported from timm.models.resnet (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/resnet_* and test/references/timm--resnet*.

@concrete struct BasicBlock
    conv1
    bn1
    act1
    conv2
    bn2
    se
    act2
    downsample
    drop_path
end
Flux.@layer BasicBlock

const BASIC_EXPANSION = 1

function BasicBlock(inplanes, planes; stride = 1, downsample = nothing, cardinality = 1,
        base_width = 64, reduce_first = 1, act_layer = relu, attn_layer = nothing,
        drop_path_rate = 0.0)
    cardinality == 1 || error("BasicBlock only supports cardinality of 1")
    base_width == 64 || error("BasicBlock does not support changing base width")
    first_planes = planes ÷ reduce_first
    outplanes = planes * BASIC_EXPANSION
    return BasicBlock(
        Conv((3, 3), inplanes => first_planes; stride, pad = 1, bias = false),
        BatchNorm(first_planes), act_layer,
        Conv((3, 3), first_planes => outplanes; pad = 1, bias = false),
        BatchNorm(outplanes), create_attn(attn_layer, outplanes), act_layer, downsample,
        drop_path(drop_path_rate))
end

function (b::BasicBlock)(x)
    shortcut = x
    x = b.act1(b.bn1(b.conv1(x)))
    x = b.bn2(b.conv2(x))
    b.se === nothing || (x = b.se(x))
    b.drop_path === nothing || (x = b.drop_path(x))
    b.downsample === nothing || (shortcut = b.downsample(shortcut))
    return b.act2(x .+ shortcut)
end

@concrete struct Bottleneck
    conv1
    bn1
    act1
    conv2
    bn2
    act2
    conv3
    bn3
    se
    act3
    downsample
    drop_path
end
Flux.@layer Bottleneck

const BOTTLENECK_EXPANSION = 4

function Bottleneck(inplanes, planes; stride = 1, downsample = nothing, cardinality = 1,
        base_width = 64, reduce_first = 1, act_layer = relu, attn_layer = nothing,
        drop_path_rate = 0.0)
    width = floor(Int, planes * (base_width / 64)) * cardinality
    first_planes = width ÷ reduce_first
    outplanes = planes * BOTTLENECK_EXPANSION
    return Bottleneck(Conv((1, 1), inplanes => first_planes; bias = false),
        BatchNorm(first_planes), act_layer,
        Conv((3, 3), first_planes => width; stride,
            pad = 1, groups = cardinality, bias = false),
        BatchNorm(width), act_layer,
        Conv((1, 1), width => outplanes; bias = false), BatchNorm(outplanes),
        create_attn(attn_layer, outplanes), act_layer, downsample,
        drop_path(drop_path_rate))
end

function (b::Bottleneck)(x)
    shortcut = x
    x = b.act1(b.bn1(b.conv1(x)))
    x = b.act2(b.bn2(b.conv2(x)))
    x = b.bn3(b.conv3(x))
    b.se === nothing || (x = b.se(x))
    b.drop_path === nothing || (x = b.drop_path(x))
    b.downsample === nothing || (shortcut = b.downsample(shortcut))
    return b.act3(x .+ shortcut)
end

# Block name => (constructor, expansion, accepted `block_args`); other files add blocks.
const RESNET_BLOCKS = Dict{String, Any}(
    "BasicBlock" => (BasicBlock, BASIC_EXPANSION, (:attn_layer,)),
    "Bottleneck" => (Bottleneck, BOTTLENECK_EXPANSION, (:attn_layer,)))

function block_type(name)
    return get(RESNET_BLOCKS, String(name)) do
        return error("unknown ResNet block $name")
    end
end

# `block_args` is a NamedTuple (from config.json), a Dict or `nothing`.
block_kwargs(::Nothing) = (;)
block_kwargs(args::NamedTuple) = args
block_kwargs(args::AbstractDict) = (; (Symbol(k) => v for (k, v) in args)...)

function downsample_conv(in_channels, out_channels, kernel_size; stride = 1)
    kernel_size = stride == 1 ? 1 : kernel_size
    p = get_padding(kernel_size, stride)
    return Chain(
        Conv((kernel_size, kernel_size), in_channels => out_channels;
            stride, pad = p, bias = false),
        BatchNorm(out_channels))
end

function downsample_avg(in_channels, out_channels; stride = 1)
    pool = stride == 1 ? identity : x -> avg_pool_ceil(x, 2, stride)
    return Chain(pool, Conv((1, 1), in_channels => out_channels; bias = false), BatchNorm(out_channels))
end

function make_blocks(
        block_fn, expansion, channels, block_repeats, inplanes; reduce_first = 1,
        down_kernel_size = 1, avg_down = false, drop_path_rate = 0.0, kwargs...)
    stages = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    for (stage_idx, (planes, num_blocks)) in enumerate(zip(channels, block_repeats))
        stride = stage_idx == 1 ? 1 : 2
        downsample = nothing
        if stride != 1 || inplanes != planes * expansion
            downsample = avg_down ? downsample_avg(inplanes, planes * expansion; stride) :
                         downsample_conv(inplanes, planes * expansion, down_kernel_size; stride)
        end
        blocks = []
        for block_idx in 1:num_blocks
            block_dpr = drop_path_rate * net_block_idx / max(net_num_blocks - 1, 1)
            push!(blocks,
                block_fn(inplanes, planes; stride = block_idx == 1 ? stride : 1,
                    downsample = block_idx == 1 ? downsample : nothing,
                    reduce_first, drop_path_rate = block_dpr, kwargs...))
            inplanes = planes * expansion
            net_block_idx += 1
        end
        push!(stages, Chain([blocks...]))
    end
    return stages
end

@concrete struct ResNet
    conv1
    bn1
    act1
    maxpool
    layer1
    layer2
    layer3
    layer4
    global_pool
    head_drop
    fc
end
Flux.@layer ResNet

"""
    resnet(; block, layers, num_classes = 1000, in_chans = 3, kwargs...)

Port of timm's `ResNet` (ResNet, ResNeXt, Wide ResNet, SE-ResNet, ECA-ResNet, Res2Net and
the c/d/s/t stem variants). Keyword arguments are those of the timm class.
`block_args` is a `NamedTuple` of keywords for the blocks: `attn_layer = "se"` or `"eca"`
for `BasicBlock` and `Bottleneck`, `scale` for `Bottle2neck`.
"""
function resnet(; block, layers, num_classes = 1000, in_chans = 3, output_stride = 32,
        global_pool = "avg", cardinality = 1, base_width = 64, stem_width = 64,
        stem_type = "", replace_stem_pool = false, block_reduce_first = 1,
        down_kernel_size = 1, avg_down = false, channels = (64, 128, 256, 512),
        act_layer = "relu", norm_layer = "batchnorm2d", aa_layer = nothing,
        drop_rate = 0.0, drop_path_rate = 0.0, drop_block_rate = 0.0,
        zero_init_last = true, block_args = nothing)
    check_unsupported(
        "ResNet"; output_stride = (output_stride, 32), global_pool = (global_pool, "avg"),
        replace_stem_pool = (replace_stem_pool, false), act_layer = (act_layer, "relu"),
        norm_layer = (norm_layer, "batchnorm2d"), aa_layer = (aa_layer, nothing),
        drop_block_rate = (drop_block_rate, 0.0))
    length(layers) == length(channels) == 4 || error("ResNet ports have exactly 4 stages")
    block_fn, expansion, accepted = block_type(block)
    block_args = block_kwargs(block_args)
    for k in keys(block_args)
        k in accepted ||
            error("`block_args` entry `$k` is not supported by the port of block $block")
    end
    deep_stem = occursin("deep", stem_type)
    inplanes = deep_stem ? stem_width * 2 : 64
    if deep_stem
        c1, c2 = occursin("tiered", stem_type) ? (3 * (stem_width ÷ 4), stem_width) :
                 (stem_width, stem_width)
        conv1 = Chain(Conv((3, 3), in_chans => c1; stride = 2, pad = 1, bias = false),
            BatchNorm(c1), relu,
            Conv((3, 3), c1 => c2; pad = 1, bias = false), BatchNorm(c2), relu,
            Conv((3, 3), c2 => inplanes; pad = 1, bias = false))
    else
        conv1 = Conv((7, 7), in_chans => inplanes; stride = 2, pad = 3, bias = false)
    end
    stages = make_blocks(
        block_fn, expansion, channels, layers, inplanes; cardinality, base_width,
        reduce_first = block_reduce_first, down_kernel_size, avg_down, drop_path_rate,
        block_args...)
    num_features = channels[end] * expansion
    fc = num_classes > 0 ? Dense(num_features => num_classes) : identity
    return ResNet(
        conv1, BatchNorm(inplanes), relu, MaxPool((3, 3); stride = 2, pad = 1), stages...,
        select_adaptive_pool2d(; flatten = true), drop_rate > 0 ? Dropout(drop_rate) :
                                                  identity, fc)
end

function forward_features(m::ResNet, x)
    x = m.maxpool(m.act1(m.bn1(m.conv1(x))))
    return m.layer4(m.layer3(m.layer2(m.layer1(x))))
end

function forward_head(m::ResNet, x; pre_logits::Bool = false)
    x = m.head_drop(m.global_pool(x))
    return pre_logits ? x : m.fc(x)
end

(m::ResNet)(x) = forward_head(m, forward_features(m, x))

const RESNET_REFERENCES = Dict(
    "resnet18" => ["timm/resnet18.a1_in1k" => "491b427b45c94c7fb0e78b5474cc919aff584bbf"],
    "resnet26t" =>
        ["timm/resnet26t.ra2_in1k" => "3dbfc891fe1d40740d8d5586227bd64c3664383c"],
    "resnet50" => ["timm/resnet50.tv2_in1k" => "10eea9779fc8a753af919714889bd7f60fba412d"],
    "resnet50d" =>
        ["timm/resnet50d.ra2_in1k" => "ea1a9b5ed97c7ca1b1401a81bbf9a19c9044797b"],
    "resnext50_32x4d" =>
        ["timm/resnext50_32x4d.a1h_in1k" => "cc736cb0940158d05966023116d90fca4c38dae3"],
    "wide_resnet50_2" =>
        ["timm/wide_resnet50_2.tv2_in1k" => "ffb9dc0ac74364ad268e066aa4aea75089bf383d"],
    "seresnext26d_32x4d" =>
        ["timm/seresnext26d_32x4d.bt_in1k" => "1214c4b9e7470f613b1ffb1f3c0243ce2736b925"],
    "ecaresnet26t" =>
        ["timm/ecaresnet26t.ra2_in1k" => "ab90506b283999a8fe6f795c3150e32c7688d791"]
)

let deep = (stem_width = 32, stem_type = "deep"),
    deep_avg = (stem_width = 32, stem_type = "deep", avg_down = true),
    tiered_avg = (stem_width = 32, stem_type = "deep_tiered", avg_down = true),
    basic = (; block = :BasicBlock), bottleneck = (; block = :Bottleneck),
    se = (; block_args = (attn_layer = "se",)), eca = (; block_args = (attn_layer = "eca",))

    for (name, args) in [
        "resnet10t" => (; basic..., layers = (1, 1, 1, 1), tiered_avg...),
        "resnet14t" => (; bottleneck..., layers = (1, 1, 1, 1), tiered_avg...),
        "resnet18" => (; basic..., layers = (2, 2, 2, 2)),
        "resnet18d" => (; basic..., layers = (2, 2, 2, 2), deep_avg...),
        "resnet26" => (; bottleneck..., layers = (2, 2, 2, 2)),
        "resnet26d" => (; bottleneck..., layers = (2, 2, 2, 2), deep_avg...),
        "resnet26t" => (; bottleneck..., layers = (2, 2, 2, 2), tiered_avg...),
        "resnet34" => (; basic..., layers = (3, 4, 6, 3)),
        "resnet34d" => (; basic..., layers = (3, 4, 6, 3), deep_avg...),
        "resnet50" => (; bottleneck..., layers = (3, 4, 6, 3)),
        "resnet50c" => (; bottleneck..., layers = (3, 4, 6, 3), deep...),
        "resnet50d" => (; bottleneck..., layers = (3, 4, 6, 3), deep_avg...),
        "resnet50s" =>
            (; bottleneck..., layers = (3, 4, 6, 3), stem_width = 64, stem_type = "deep"),
        "resnet50t" => (; bottleneck..., layers = (3, 4, 6, 3), tiered_avg...),
        "resnet101" => (; bottleneck..., layers = (3, 4, 23, 3)),
        "resnet101c" => (; bottleneck..., layers = (3, 4, 23, 3), deep...),
        "resnet101d" => (; bottleneck..., layers = (3, 4, 23, 3), deep_avg...),
        "resnet101s" =>
            (; bottleneck..., layers = (3, 4, 23, 3), stem_width = 64, stem_type = "deep"),
        "resnet152" => (; bottleneck..., layers = (3, 8, 36, 3)),
        "resnet152c" => (; bottleneck..., layers = (3, 8, 36, 3), deep...),
        "resnet152d" => (; bottleneck..., layers = (3, 8, 36, 3), deep_avg...),
        "resnet152s" =>
            (; bottleneck..., layers = (3, 8, 36, 3), stem_width = 64, stem_type = "deep"),
        "resnet200d" => (; bottleneck..., layers = (3, 24, 36, 3), deep_avg...),
        "wide_resnet50_2" => (; bottleneck..., layers = (3, 4, 6, 3), base_width = 128),
        "wide_resnet101_2" => (; bottleneck..., layers = (3, 4, 23, 3), base_width = 128),
        "resnext50_32x4d" =>
            (; bottleneck..., layers = (3, 4, 6, 3), cardinality = 32, base_width = 4),
        "resnext50d_32x4d" => (; bottleneck..., layers = (3, 4, 6, 3),
            cardinality = 32, base_width = 4, deep_avg...),
        "resnext101_32x4d" =>
            (; bottleneck..., layers = (3, 4, 23, 3), cardinality = 32, base_width = 4),
        "resnext101_32x8d" =>
            (; bottleneck..., layers = (3, 4, 23, 3), cardinality = 32, base_width = 8),
        "resnext101_32x16d" =>
            (; bottleneck..., layers = (3, 4, 23, 3), cardinality = 32, base_width = 16),
        "resnext101_32x32d" =>
            (; bottleneck..., layers = (3, 4, 23, 3), cardinality = 32, base_width = 32),
        "resnext101_64x4d" =>
            (; bottleneck..., layers = (3, 4, 23, 3), cardinality = 64, base_width = 4),
        "seresnet18" => (; basic..., layers = (2, 2, 2, 2), se...),
        "seresnet34" => (; basic..., layers = (3, 4, 6, 3), se...),
        "seresnet50" => (; bottleneck..., layers = (3, 4, 6, 3), se...),
        "seresnet50t" => (; bottleneck..., layers = (3, 4, 6, 3), tiered_avg..., se...),
        "seresnet101" => (; bottleneck..., layers = (3, 4, 23, 3), se...),
        "seresnet152" => (; bottleneck..., layers = (3, 8, 36, 3), se...),
        "seresnet152d" => (; bottleneck..., layers = (3, 8, 36, 3), deep_avg..., se...),
        "seresnet200d" => (; bottleneck..., layers = (3, 24, 36, 3), deep_avg..., se...),
        "seresnet269d" => (; bottleneck..., layers = (3, 30, 48, 8), deep_avg..., se...),
        "seresnext26d_32x4d" => (; bottleneck..., layers = (2, 2, 2, 2),
            cardinality = 32, base_width = 4, deep_avg..., se...),
        "seresnext26t_32x4d" => (; bottleneck..., layers = (2, 2, 2, 2),
            cardinality = 32, base_width = 4, tiered_avg..., se...),
        "seresnext50_32x4d" => (; bottleneck..., layers = (3, 4, 6, 3),
            cardinality = 32, base_width = 4, se...),
        "seresnext101_32x4d" => (; bottleneck..., layers = (3, 4, 23, 3),
            cardinality = 32, base_width = 4, se...),
        "seresnext101_32x8d" => (; bottleneck..., layers = (3, 4, 23, 3),
            cardinality = 32, base_width = 8, se...),
        "seresnext101d_32x8d" => (; bottleneck..., layers = (3, 4, 23, 3),
            cardinality = 32, base_width = 8, deep_avg..., se...),
        "seresnext101_64x4d" => (; bottleneck..., layers = (3, 4, 23, 3),
            cardinality = 64, base_width = 4, se...),
        "senet154" => (; bottleneck..., layers = (3, 8, 36, 3), cardinality = 64,
            base_width = 4, stem_type = "deep", down_kernel_size = 3,
            block_reduce_first = 2, se...),
        "ecaresnet26t" => (; bottleneck..., layers = (2, 2, 2, 2), tiered_avg..., eca...),
        "ecaresnet50d" => (; bottleneck..., layers = (3, 4, 6, 3), deep_avg..., eca...),
        "ecaresnet50t" => (; bottleneck..., layers = (3, 4, 6, 3), tiered_avg..., eca...),
        "ecaresnetlight" => (; bottleneck..., layers = (1, 1, 11, 3),
            stem_width = 32, avg_down = true, eca...),
        "ecaresnet101d" => (; bottleneck..., layers = (3, 4, 23, 3), deep_avg..., eca...),
        "ecaresnet200d" => (; bottleneck..., layers = (3, 24, 36, 3), deep_avg..., eca...),
        "ecaresnet269d" => (; bottleneck..., layers = (3, 30, 48, 8), deep_avg..., eca...),
        "ecaresnext26t_32x4d" => (; bottleneck..., layers = (2, 2, 2, 2),
            cardinality = 32, base_width = 4, tiered_avg..., eca...),
        "ecaresnext50t_32x4d" => (; bottleneck..., layers = (2, 2, 2, 2),
            cardinality = 32, base_width = 4, tiered_avg..., eca...)
    ]
        register!(ArchSpec(:timm, name, resnet; defaults = args,
            references = get(RESNET_REFERENCES, name, [])))
    end
end
