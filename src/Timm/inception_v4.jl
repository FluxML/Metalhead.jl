# Ported from timm.models.inception_v4 (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/inception_v4 and test/references/timm--inception_v4*.

@concrete struct InceptionV4Mixed3a
    maxpool
    conv
end
Flux.@layer InceptionV4Mixed3a

function InceptionV4Mixed3a()
    return InceptionV4Mixed3a(
        MaxPool((3, 3); stride = 2), inception_conv(64, 96, 3; stride = 2))
end

function (m::InceptionV4Mixed3a)(x)
    return cat(m.maxpool(x), m.conv(x); dims = 3)
end

@concrete struct InceptionV4Mixed4a
    branch0
    branch1
end
Flux.@layer InceptionV4Mixed4a

function InceptionV4Mixed4a()
    return InceptionV4Mixed4a(
        Chain(inception_conv(160, 64, 1), inception_conv(64, 96, 3)),
        Chain(inception_conv(160, 64, 1),
            inception_conv(64, 64, (1, 7); padding = (0, 3)),
            inception_conv(64, 64, (7, 1); padding = (3, 0)),
            inception_conv(64, 96, (3, 3))))
end

(m::InceptionV4Mixed4a)(x) = cat(m.branch0(x), m.branch1(x); dims = 3)

@concrete struct InceptionV4Mixed5a
    conv
    maxpool
end
Flux.@layer InceptionV4Mixed5a

function InceptionV4Mixed5a()
    return InceptionV4Mixed5a(
        inception_conv(192, 192, 3; stride = 2), MaxPool((3, 3); stride = 2))
end

(m::InceptionV4Mixed5a)(x) = cat(m.conv(x), m.maxpool(x); dims = 3)

@concrete struct InceptionV4A
    branch0
    branch1
    branch2
    branch3
end
Flux.@layer InceptionV4A

function InceptionV4A()
    return InceptionV4A(
        inception_conv(384, 96, 1),
        Chain(inception_conv(384, 64, 1), inception_conv(64, 96, 3; padding = 1)),
        Chain(inception_conv(384, 64, 1), inception_conv(64, 96, 3; padding = 1),
            inception_conv(96, 96, 3; padding = 1)),
        Chain(avg_pool3_exclude_pad, inception_conv(384, 96, 1)))
end

function (m::InceptionV4A)(x)
    return cat(m.branch0(x), m.branch1(x), m.branch2(x), m.branch3(x); dims = 3)
end

@concrete struct InceptionV4ReductionA
    branch0
    branch1
    branch2
end
Flux.@layer InceptionV4ReductionA

function InceptionV4ReductionA()
    return InceptionV4ReductionA(
        inception_conv(384, 384, 3; stride = 2),
        Chain(inception_conv(384, 192, 1), inception_conv(192, 224, 3; padding = 1),
            inception_conv(224, 256, 3; stride = 2)),
        MaxPool((3, 3); stride = 2))
end

(m::InceptionV4ReductionA)(x) = cat(m.branch0(x), m.branch1(x), m.branch2(x); dims = 3)

@concrete struct InceptionV4B
    branch0
    branch1
    branch2
    branch3
end
Flux.@layer InceptionV4B

function InceptionV4B()
    return InceptionV4B(
        inception_conv(1024, 384, 1),
        Chain(inception_conv(1024, 192, 1),
            inception_conv(192, 224, (1, 7); padding = (0, 3)),
            inception_conv(224, 256, (7, 1); padding = (3, 0))),
        Chain(inception_conv(1024, 192, 1),
            inception_conv(192, 192, (7, 1); padding = (3, 0)),
            inception_conv(192, 224, (1, 7); padding = (0, 3)),
            inception_conv(224, 224, (7, 1); padding = (3, 0)),
            inception_conv(224, 256, (1, 7); padding = (0, 3))),
        Chain(avg_pool3_exclude_pad, inception_conv(1024, 128, 1)))
end

function (m::InceptionV4B)(x)
    return cat(m.branch0(x), m.branch1(x), m.branch2(x), m.branch3(x); dims = 3)
end

@concrete struct InceptionV4ReductionB
    branch0
    branch1
    branch2
end
Flux.@layer InceptionV4ReductionB

function InceptionV4ReductionB()
    return InceptionV4ReductionB(
        Chain(inception_conv(1024, 192, 1), inception_conv(192, 192, 3; stride = 2)),
        Chain(inception_conv(1024, 256, 1),
            inception_conv(256, 256, (1, 7); padding = (0, 3)),
            inception_conv(256, 320, (7, 1); padding = (3, 0)),
            inception_conv(320, 320, 3; stride = 2)),
        MaxPool((3, 3); stride = 2))
end

(m::InceptionV4ReductionB)(x) = cat(m.branch0(x), m.branch1(x), m.branch2(x); dims = 3)

@concrete struct InceptionV4C
    branch0
    branch1_0
    branch1_1a
    branch1_1b
    branch2_0
    branch2_1
    branch2_2
    branch2_3a
    branch2_3b
    branch3
end
Flux.@layer InceptionV4C

function InceptionV4C()
    return InceptionV4C(
        inception_conv(1536, 256, 1),
        inception_conv(1536, 384, 1),
        inception_conv(384, 256, (1, 3); padding = (0, 1)),
        inception_conv(384, 256, (3, 1); padding = (1, 0)),
        inception_conv(1536, 384, 1),
        inception_conv(384, 448, (3, 1); padding = (1, 0)),
        inception_conv(448, 512, (1, 3); padding = (0, 1)),
        inception_conv(512, 256, (1, 3); padding = (0, 1)),
        inception_conv(512, 256, (3, 1); padding = (1, 0)),
        Chain(avg_pool3_exclude_pad, inception_conv(1536, 256, 1)))
end

function (m::InceptionV4C)(x)
    x0 = m.branch0(x)
    x1_0 = m.branch1_0(x)
    x1 = cat(m.branch1_1a(x1_0), m.branch1_1b(x1_0); dims = 3)
    x2_2 = m.branch2_2(m.branch2_1(m.branch2_0(x)))
    x2 = cat(m.branch2_3a(x2_2), m.branch2_3b(x2_2); dims = 3)
    return cat(x0, x1, x2, m.branch3(x); dims = 3)
end

@concrete struct InceptionV4
    features
    global_pool
    head_drop
    last_linear
end
Flux.@layer InceptionV4

"""
    inception_v4(; num_classes = 1000, in_chans = 3, drop_rate = 0.0)

Port of timm's `InceptionV4`. Keyword arguments are those of the timm class.
"""
function inception_v4(; num_classes = 1000, in_chans = 3, output_stride = 32,
        drop_rate = 0.0, global_pool = "avg", norm_layer = "batchnorm2d", norm_eps = 1e-3,
        act_layer = "relu")
    check_unsupported("InceptionV4"; output_stride = (output_stride, 32),
        global_pool = (global_pool, "avg"), norm_layer = (norm_layer, "batchnorm2d"),
        norm_eps = (norm_eps, 1e-3), act_layer = (act_layer, "relu"))
    features = Chain(
        inception_conv(in_chans, 32, 3; stride = 2), inception_conv(32, 32, 3),
        inception_conv(32, 64, 3; padding = 1),
        InceptionV4Mixed3a(), InceptionV4Mixed4a(), InceptionV4Mixed5a(),
        [InceptionV4A() for _ in 1:4]..., InceptionV4ReductionA(),
        [InceptionV4B() for _ in 1:7]..., InceptionV4ReductionB(),
        [InceptionV4C() for _ in 1:3]...)
    pool, head_drop, last_linear = create_classifier(1536, num_classes; drop_rate)
    return InceptionV4(features, pool, head_drop, last_linear)
end

forward_features(m::InceptionV4, x) = m.features(x)

function forward_head(m::InceptionV4, x; pre_logits::Bool = false)
    x = m.head_drop(m.global_pool(x))
    return pre_logits ? x : m.last_linear(x)
end

(m::InceptionV4)(x) = forward_head(m, forward_features(m, x))

# The `features` entries ending a stage (timm's `feature_info`), and their strides.
const INCEPTION_V4_STAGE_ENDS = (3, 4, 10, 18, 22)
const INCEPTION_V4_REDUCTIONS = (2, 4, 8, 16, 32)

# `norm` is accepted for timm's signature but there is no norm.
function forward_intermediates(m::InceptionV4, x; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, output_fmt = "NCHW", intermediates_only::Bool = false)
    check_output_fmt(output_fmt)
    ends = INCEPTION_V4_STAGE_ENDS
    take, max_index = feature_take_indices(length(ends), indices)
    stop = stop_early ? ends[max_index] : length(m.features)
    outs = stage_outputs(m.features.layers, x, stop)
    intermediates = [outs[ends[i]] for i in take]
    intermediates_only && return intermediates
    return last(outs), intermediates
end

# Output channels of the blocks that end a stage.
inception_v4_chs(c::Chain) = inception_v4_chs(c[end])
inception_v4_chs(b::BatchNorm) = b.chs
inception_v4_chs(m::InceptionV4Mixed3a) = size(m.conv[:conv].weight, 3) + m.conv[:bn].chs
function inception_v4_chs(m::Union{InceptionV4A, InceptionV4B})
    return sum(inception_v4_chs, (m.branch0, m.branch1, m.branch2, m.branch3))
end
function inception_v4_chs(m::InceptionV4C)
    return sum(inception_v4_chs,
        (m.branch0, m.branch1_1a, m.branch1_1b, m.branch2_3a, m.branch2_3b, m.branch3))
end

function feature_info(m::InceptionV4)
    return [(; num_chs = inception_v4_chs(m.features[e]), reduction,
                module_name = "features.$(e - 1)")
            for (e, reduction) in zip(INCEPTION_V4_STAGE_ENDS, INCEPTION_V4_REDUCTIONS)]
end

register!(ArchSpec(TIMM, "inception_v4", inception_v4; family = "inception",
    references = ["timm/inception_v4.tf_in1k" =>
        "ad5d294cda312745a9afa433e45d9ccd956a1548"]))
