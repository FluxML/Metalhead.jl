# Ported from timm.models.inception_resnet_v2 (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/inception_resnet_v2 and
# test/references/timm--inception_resnet_v2*.

@concrete struct InceptionResNetV2Mixed5b
    branch0
    branch1
    branch2
    branch3
end
Flux.@layer InceptionResNetV2Mixed5b

function InceptionResNetV2Mixed5b()
    return InceptionResNetV2Mixed5b(
        inception_conv(192, 96, 1),
        Chain(inception_conv(192, 48, 1), inception_conv(48, 64, 5; padding = 2)),
        Chain(inception_conv(192, 64, 1), inception_conv(64, 96, 3; padding = 1),
            inception_conv(96, 96, 3; padding = 1)),
        Chain(avg_pool3_exclude_pad, inception_conv(192, 64, 1)))
end

function (m::InceptionResNetV2Mixed5b)(x)
    return cat(m.branch0(x), m.branch1(x), m.branch2(x), m.branch3(x); dims = 3)
end

# `out * scale + x` of the residual blocks, with the Python float cast to the eltype of x.
scaled_residual(out, scale, x) = out .* eltype(x)(scale) .+ x

@concrete struct InceptionResNetV2Block35
    scale
    branch0
    branch1
    branch2
    conv2d
    act
end
Flux.@layer InceptionResNetV2Block35

function InceptionResNetV2Block35(; scale = 1.0)
    return InceptionResNetV2Block35(scale,
        inception_conv(320, 32, 1),
        Chain(inception_conv(320, 32, 1), inception_conv(32, 32, 3; padding = 1)),
        Chain(inception_conv(320, 32, 1), inception_conv(32, 48, 3; padding = 1),
            inception_conv(48, 64, 3; padding = 1)),
        Conv((1, 1), 128 => 320), relu)
end

function (m::InceptionResNetV2Block35)(x)
    out = cat(m.branch0(x), m.branch1(x), m.branch2(x); dims = 3)
    return m.act(scaled_residual(m.conv2d(out), m.scale, x))
end

@concrete struct InceptionResNetV2Mixed6a
    branch0
    branch1
    branch2
end
Flux.@layer InceptionResNetV2Mixed6a

function InceptionResNetV2Mixed6a()
    return InceptionResNetV2Mixed6a(
        inception_conv(320, 384, 3; stride = 2),
        Chain(inception_conv(320, 256, 1), inception_conv(256, 256, 3; padding = 1),
            inception_conv(256, 384, 3; stride = 2)),
        MaxPool((3, 3); stride = 2))
end

function (m::InceptionResNetV2Mixed6a)(x)
    return cat(m.branch0(x), m.branch1(x), m.branch2(x); dims = 3)
end

@concrete struct InceptionResNetV2Block17
    scale
    branch0
    branch1
    conv2d
    act
end
Flux.@layer InceptionResNetV2Block17

function InceptionResNetV2Block17(; scale = 1.0)
    return InceptionResNetV2Block17(scale,
        inception_conv(1088, 192, 1),
        Chain(inception_conv(1088, 128, 1),
            inception_conv(128, 160, (1, 7); padding = (0, 3)),
            inception_conv(160, 192, (7, 1); padding = (3, 0))),
        Conv((1, 1), 384 => 1088), relu)
end

function (m::InceptionResNetV2Block17)(x)
    out = cat(m.branch0(x), m.branch1(x); dims = 3)
    return m.act(scaled_residual(m.conv2d(out), m.scale, x))
end

@concrete struct InceptionResNetV2Mixed7a
    branch0
    branch1
    branch2
    branch3
end
Flux.@layer InceptionResNetV2Mixed7a

function InceptionResNetV2Mixed7a()
    return InceptionResNetV2Mixed7a(
        Chain(inception_conv(1088, 256, 1), inception_conv(256, 384, 3; stride = 2)),
        Chain(inception_conv(1088, 256, 1), inception_conv(256, 288, 3; stride = 2)),
        Chain(inception_conv(1088, 256, 1), inception_conv(256, 288, 3; padding = 1),
            inception_conv(288, 320, 3; stride = 2)),
        MaxPool((3, 3); stride = 2))
end

function (m::InceptionResNetV2Mixed7a)(x)
    return cat(m.branch0(x), m.branch1(x), m.branch2(x), m.branch3(x); dims = 3)
end

@concrete struct InceptionResNetV2Block8
    scale
    branch0
    branch1
    conv2d
    relu
end
Flux.@layer InceptionResNetV2Block8

function InceptionResNetV2Block8(; scale = 1.0, no_relu = false)
    return InceptionResNetV2Block8(scale,
        inception_conv(2080, 192, 1),
        Chain(inception_conv(2080, 192, 1),
            inception_conv(192, 224, (1, 3); padding = (0, 1)),
            inception_conv(224, 256, (3, 1); padding = (1, 0))),
        Conv((1, 1), 448 => 2080), no_relu ? nothing : relu)
end

function (m::InceptionResNetV2Block8)(x)
    out = cat(m.branch0(x), m.branch1(x); dims = 3)
    out = scaled_residual(m.conv2d(out), m.scale, x)
    return m.relu === nothing ? out : m.relu(out)
end

@concrete struct InceptionResNetV2
    conv2d_1a
    conv2d_2a
    conv2d_2b
    maxpool_3a
    conv2d_3b
    conv2d_4a
    maxpool_5a
    mixed_5b
    repeat
    mixed_6a
    repeat_1
    mixed_7a
    repeat_2
    block8
    conv2d_7b
    global_pool
    head_drop
    classif
end
Flux.@layer InceptionResNetV2

"""
    inception_resnet_v2(; num_classes = 1000, in_chans = 3, drop_rate = 0.0)

Port of timm's `InceptionResnetV2`. Keyword arguments are those of the timm class.
"""
function inception_resnet_v2(; num_classes = 1000, in_chans = 3, drop_rate = 0.0,
        output_stride = 32, global_pool = "avg", norm_layer = "batchnorm2d",
        norm_eps = 1e-3, act_layer = "relu")
    check_unsupported("InceptionResNetV2"; output_stride = (output_stride, 32),
        global_pool = (global_pool, "avg"), norm_layer = (norm_layer, "batchnorm2d"),
        norm_eps = (norm_eps, 1e-3), act_layer = (act_layer, "relu"))
    pool, head_drop, classif = create_classifier(1536, num_classes; drop_rate)
    return InceptionResNetV2(
        inception_conv(in_chans, 32, 3; stride = 2), inception_conv(32, 32, 3),
        inception_conv(32, 64, 3; padding = 1), MaxPool((3, 3); stride = 2),
        inception_conv(64, 80, 1), inception_conv(80, 192, 3), MaxPool((3, 3); stride = 2),
        InceptionResNetV2Mixed5b(),
        Chain([InceptionResNetV2Block35(; scale = 0.17) for _ in 1:10]),
        InceptionResNetV2Mixed6a(),
        Chain([InceptionResNetV2Block17(; scale = 0.10) for _ in 1:20]),
        InceptionResNetV2Mixed7a(),
        Chain([InceptionResNetV2Block8(; scale = 0.20) for _ in 1:9]),
        InceptionResNetV2Block8(; no_relu = true), inception_conv(2080, 1536, 1),
        pool, head_drop, classif)
end

function forward_features(m::InceptionResNetV2, x)
    x = m.conv2d_2b(m.conv2d_2a(m.conv2d_1a(x)))
    x = m.conv2d_4a(m.conv2d_3b(m.maxpool_3a(x)))
    x = m.repeat(m.mixed_5b(m.maxpool_5a(x)))
    x = m.repeat_1(m.mixed_6a(x))
    x = m.repeat_2(m.mixed_7a(x))
    return m.conv2d_7b(m.block8(x))
end

function forward_head(m::InceptionResNetV2, x; pre_logits::Bool = false)
    x = m.head_drop(m.global_pool(x))
    return pre_logits ? x : m.classif(x)
end

(m::InceptionResNetV2)(x) = forward_head(m, forward_features(m, x))

register!(ArchSpec(TIMM, "inception_resnet_v2", inception_resnet_v2;
    references = ["timm/inception_resnet_v2.tf_in1k" =>
        "836b07d7d599da247de92f148c7a07fda18afced"]))
