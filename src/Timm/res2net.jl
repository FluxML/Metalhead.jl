# Ported from timm.models.res2net (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/res2ne* and test/references/timm--res2net*.

@concrete struct Bottle2neck
    conv1
    bn1
    convs
    bns
    pool
    conv3
    bn3
    relu
    downsample
    scale
    width
    is_first
end
Flux.@layer Bottle2neck

const BOTTLE2NECK_EXPANSION = 4

block_out_chs(b::Bottle2neck) = b.bn3.chs

# Like timm, ignores the ResNet block arguments it has no use for (`reduce_first`,
# `drop_path_rate`) and, unlike Bottleneck, takes no attention layer.
function Bottle2neck(inplanes, planes; stride = 1, downsample = nothing, cardinality = 1,
        base_width = 26, scale = 4, act_layer = relu, kwargs...)
    width = floor(Int, planes * (base_width / 64)) * cardinality
    outplanes = planes * BOTTLE2NECK_EXPANSION
    num_scales = max(1, scale - 1)
    is_first = stride > 1 || downsample !== nothing
    convs = [init_layer(
                 Conv, (3, 3), width => width; stride, pad = 1, groups = cardinality,
                 bias = false) for _ in 1:num_scales]
    return Bottle2neck(
        init_layer(Conv, (1, 1), inplanes => width * scale; bias = false),
        init_layer(BatchNorm, width * scale),
        convs, [init_layer(BatchNorm, width) for _ in 1:num_scales],
        is_first ? MeanPool((3, 3); stride, pad = 1) : nothing,
        init_layer(Conv, (1, 1), width * scale => outplanes; bias = false),
        init_layer(BatchNorm, outplanes),
        act_layer, downsample, scale, width, is_first)
end

function (b::Bottle2neck)(x)
    out = b.relu(b.bn1(b.conv1(x)))
    # copies, not the views of `chunk`, which GPU convolutions don't take
    spx = map(r -> out[:, :, r, :], split_ranges(ntuple(_ -> b.width, b.scale)))
    sp = spx[1]
    spo = ()
    for (i, (conv, bn)) in enumerate(zip(b.convs, b.bns))
        sp = i == 1 || b.is_first ? spx[i] : sp + spx[i]
        sp = b.relu(bn(conv(sp)))
        spo = (spo..., sp)
    end
    if b.scale > 1
        spo = (spo..., b.pool === nothing ? spx[end] : b.pool(spx[end]))
    end
    x_out = b.bn3(b.conv3(cat(spo...; dims = 3)))
    shortcut = b.downsample === nothing ? x : b.downsample(x)
    return b.relu(x_out .+ shortcut)
end

RESNET_BLOCKS["Bottle2neck"] = (Bottle2neck, BOTTLE2NECK_EXPANSION, (:scale,))

const RES2NET_REFERENCES = Dict(
    "res2net50_26w_4s" =>
        ["timm/res2net50_26w_4s.in1k" => "a36466d0354c9d337dbc0af329e13cb1cd7137fe"])

let deep_avg = (stem_width = 32, stem_type = "deep", avg_down = true),
    block = (; block = :Bottle2neck), layers50 = (3, 4, 6, 3), layers101 = (3, 4, 23, 3)

    for (name, args) in [
        "res2net50_26w_4s" =>
            (; block..., layers = layers50, base_width = 26, block_args = (scale = 4,)),
        "res2net101_26w_4s" =>
            (; block..., layers = layers101, base_width = 26, block_args = (scale = 4,)),
        "res2net50_26w_6s" =>
            (; block..., layers = layers50, base_width = 26, block_args = (scale = 6,)),
        "res2net50_26w_8s" =>
            (; block..., layers = layers50, base_width = 26, block_args = (scale = 8,)),
        "res2net50_48w_2s" =>
            (; block..., layers = layers50, base_width = 48, block_args = (scale = 2,)),
        "res2net50_14w_8s" =>
            (; block..., layers = layers50, base_width = 14, block_args = (scale = 8,)),
        "res2next50" => (; block..., layers = layers50, base_width = 4, cardinality = 8,
            block_args = (scale = 4,)),
        "res2net50d" => (; block..., layers = layers50, base_width = 26, deep_avg...,
            block_args = (scale = 4,)),
        "res2net101d" => (; block..., layers = layers101, base_width = 26, deep_avg...,
            block_args = (scale = 4,))
    ]
        register!(ArchSpec(TIMM, name, resnet; family = "res2net", defaults = args,
            references = get(RES2NET_REFERENCES, name, [])))
    end
end
