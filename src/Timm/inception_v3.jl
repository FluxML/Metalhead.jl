# Ported from timm.models.inception_v3 (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/inception_v3 and test/references/timm--inception_v3*.

# The `conv_block` partial of the timm Inception models: ConvNormAct with no padding and
# BatchNorm eps 1e-3. Also used by Inception-v4 and Inception-ResNet-v2.
function inception_conv(in_chs, out_chs, kernel_size; stride = 1, padding = 0)
    return conv_norm_act(in_chs, out_chs, kernel_size; stride, padding, eps = 1e-3)
end

# F.avg_pool2d(x, 3, 1, 1), which counts the padding, and the nn.AvgPool2d that doesn't.
function avg_pool3(x; count_include_pad)
    return NNlib.meanpool(x, (3, 3); stride = 1, pad = 1, count_include_pad)
end
avg_pool3_exclude_pad(x) = avg_pool3(x; count_include_pad = false)

# F.max_pool2d(x, 3, stride = 2)
max_pool3_stride2(x) = NNlib.maxpool(x, (3, 3); stride = 2)

@concrete struct InceptionV3A
    branch1x1
    branch5x5_1
    branch5x5_2
    branch3x3dbl_1
    branch3x3dbl_2
    branch3x3dbl_3
    branch_pool
end
Flux.@layer InceptionV3A

function InceptionV3A(in_channels, pool_features)
    return InceptionV3A(
        inception_conv(in_channels, 64, 1),
        inception_conv(in_channels, 48, 1), inception_conv(48, 64, 5; padding = 2),
        inception_conv(in_channels, 64, 1), inception_conv(64, 96, 3; padding = 1),
        inception_conv(96, 96, 3; padding = 1),
        inception_conv(in_channels, pool_features, 1))
end

function (m::InceptionV3A)(x)
    branch1x1 = m.branch1x1(x)
    branch5x5 = m.branch5x5_2(m.branch5x5_1(x))
    branch3x3dbl = m.branch3x3dbl_3(m.branch3x3dbl_2(m.branch3x3dbl_1(x)))
    branch_pool = m.branch_pool(avg_pool3(x; count_include_pad = true))
    return cat(branch1x1, branch5x5, branch3x3dbl, branch_pool; dims = 3)
end

@concrete struct InceptionV3B
    branch3x3
    branch3x3dbl_1
    branch3x3dbl_2
    branch3x3dbl_3
end
Flux.@layer InceptionV3B

function InceptionV3B(in_channels)
    return InceptionV3B(
        inception_conv(in_channels, 384, 3; stride = 2),
        inception_conv(in_channels, 64, 1), inception_conv(64, 96, 3; padding = 1),
        inception_conv(96, 96, 3; stride = 2))
end

function (m::InceptionV3B)(x)
    branch3x3dbl = m.branch3x3dbl_3(m.branch3x3dbl_2(m.branch3x3dbl_1(x)))
    return cat(m.branch3x3(x), branch3x3dbl, max_pool3_stride2(x); dims = 3)
end

@concrete struct InceptionV3C
    branch1x1
    branch7x7_1
    branch7x7_2
    branch7x7_3
    branch7x7dbl_1
    branch7x7dbl_2
    branch7x7dbl_3
    branch7x7dbl_4
    branch7x7dbl_5
    branch_pool
end
Flux.@layer InceptionV3C

function InceptionV3C(in_channels, channels_7x7)
    c7 = channels_7x7
    return InceptionV3C(
        inception_conv(in_channels, 192, 1),
        inception_conv(in_channels, c7, 1),
        inception_conv(c7, c7, (1, 7); padding = (0, 3)),
        inception_conv(c7, 192, (7, 1); padding = (3, 0)),
        inception_conv(in_channels, c7, 1),
        inception_conv(c7, c7, (7, 1); padding = (3, 0)),
        inception_conv(c7, c7, (1, 7); padding = (0, 3)),
        inception_conv(c7, c7, (7, 1); padding = (3, 0)),
        inception_conv(c7, 192, (1, 7); padding = (0, 3)),
        inception_conv(in_channels, 192, 1))
end

function (m::InceptionV3C)(x)
    branch1x1 = m.branch1x1(x)
    branch7x7 = m.branch7x7_3(m.branch7x7_2(m.branch7x7_1(x)))
    branch7x7dbl = m.branch7x7dbl_1(x)
    branch7x7dbl = m.branch7x7dbl_2(branch7x7dbl)
    branch7x7dbl = m.branch7x7dbl_3(branch7x7dbl)
    branch7x7dbl = m.branch7x7dbl_4(branch7x7dbl)
    branch7x7dbl = m.branch7x7dbl_5(branch7x7dbl)
    branch_pool = m.branch_pool(avg_pool3(x; count_include_pad = true))
    return cat(branch1x1, branch7x7, branch7x7dbl, branch_pool; dims = 3)
end

@concrete struct InceptionV3D
    branch3x3_1
    branch3x3_2
    branch7x7x3_1
    branch7x7x3_2
    branch7x7x3_3
    branch7x7x3_4
end
Flux.@layer InceptionV3D

function InceptionV3D(in_channels)
    return InceptionV3D(
        inception_conv(in_channels, 192, 1), inception_conv(192, 320, 3; stride = 2),
        inception_conv(in_channels, 192, 1),
        inception_conv(192, 192, (1, 7); padding = (0, 3)),
        inception_conv(192, 192, (7, 1); padding = (3, 0)),
        inception_conv(192, 192, 3; stride = 2))
end

function (m::InceptionV3D)(x)
    branch3x3 = m.branch3x3_2(m.branch3x3_1(x))
    branch7x7x3 = m.branch7x7x3_1(x)
    branch7x7x3 = m.branch7x7x3_2(branch7x7x3)
    branch7x7x3 = m.branch7x7x3_3(branch7x7x3)
    branch7x7x3 = m.branch7x7x3_4(branch7x7x3)
    return cat(branch3x3, branch7x7x3, max_pool3_stride2(x); dims = 3)
end

@concrete struct InceptionV3E
    branch1x1
    branch3x3_1
    branch3x3_2a
    branch3x3_2b
    branch3x3dbl_1
    branch3x3dbl_2
    branch3x3dbl_3a
    branch3x3dbl_3b
    branch_pool
end
Flux.@layer InceptionV3E

function InceptionV3E(in_channels)
    return InceptionV3E(
        inception_conv(in_channels, 320, 1),
        inception_conv(in_channels, 384, 1),
        inception_conv(384, 384, (1, 3); padding = (0, 1)),
        inception_conv(384, 384, (3, 1); padding = (1, 0)),
        inception_conv(in_channels, 448, 1), inception_conv(448, 384, 3; padding = 1),
        inception_conv(384, 384, (1, 3); padding = (0, 1)),
        inception_conv(384, 384, (3, 1); padding = (1, 0)),
        inception_conv(in_channels, 192, 1))
end

function (m::InceptionV3E)(x)
    branch1x1 = m.branch1x1(x)
    branch3x3 = m.branch3x3_1(x)
    branch3x3 = cat(m.branch3x3_2a(branch3x3), m.branch3x3_2b(branch3x3); dims = 3)
    branch3x3dbl = m.branch3x3dbl_2(m.branch3x3dbl_1(x))
    branch3x3dbl = cat(
        m.branch3x3dbl_3a(branch3x3dbl), m.branch3x3dbl_3b(branch3x3dbl); dims = 3)
    branch_pool = m.branch_pool(avg_pool3(x; count_include_pad = true))
    return cat(branch1x1, branch3x3, branch3x3dbl, branch_pool; dims = 3)
end

@concrete struct InceptionV3
    Conv2d_1a_3x3
    Conv2d_2a_3x3
    Conv2d_2b_3x3
    Pool1
    Conv2d_3b_1x1
    Conv2d_4a_3x3
    Pool2
    Mixed_5b
    Mixed_5c
    Mixed_5d
    Mixed_6a
    Mixed_6b
    Mixed_6c
    Mixed_6d
    Mixed_6e
    Mixed_7a
    Mixed_7b
    Mixed_7c
    global_pool
    head_drop
    fc
end
Flux.@layer InceptionV3

"""
    inception_v3(; num_classes = 1000, in_chans = 3, drop_rate = 0.0)

Port of timm's `InceptionV3` without the auxiliary classifier (`aux_logits = false`).
Keyword arguments are those of the timm class. The input must be at least 75x75.
"""
function inception_v3(; num_classes = 1000, in_chans = 3, drop_rate = 0.0,
        global_pool = "avg", aux_logits = false, norm_layer = "batchnorm2d",
        norm_eps = 1e-3, act_layer = "relu")
    check_unsupported("InceptionV3"; global_pool = (global_pool, "avg"),
        aux_logits = (aux_logits, false), norm_layer = (norm_layer, "batchnorm2d"),
        norm_eps = (norm_eps, 1e-3), act_layer = (act_layer, "relu"))
    pool, head_drop, fc = create_classifier(2048, num_classes; drop_rate)
    return InceptionV3(
        inception_conv(in_chans, 32, 3; stride = 2), inception_conv(32, 32, 3),
        inception_conv(32, 64, 3; padding = 1), MaxPool((3, 3); stride = 2),
        inception_conv(64, 80, 1), inception_conv(80, 192, 3), MaxPool((3, 3); stride = 2),
        InceptionV3A(192, 32), InceptionV3A(256, 64), InceptionV3A(288, 64),
        InceptionV3B(288),
        InceptionV3C(768, 128), InceptionV3C(768, 160), InceptionV3C(768, 160),
        InceptionV3C(768, 192),
        InceptionV3D(768), InceptionV3E(1280), InceptionV3E(2048),
        pool, head_drop, fc)
end

function forward_features(m::InceptionV3, x)
    x = m.Conv2d_1a_3x3(x)
    x = m.Conv2d_2a_3x3(x)
    x = m.Conv2d_2b_3x3(x)
    x = m.Pool1(x)
    x = m.Conv2d_3b_1x1(x)
    x = m.Conv2d_4a_3x3(x)
    x = m.Pool2(x)
    x = m.Mixed_5d(m.Mixed_5c(m.Mixed_5b(x)))
    x = m.Mixed_6e(m.Mixed_6d(m.Mixed_6c(m.Mixed_6b(m.Mixed_6a(x)))))
    return m.Mixed_7c(m.Mixed_7b(m.Mixed_7a(x)))
end

function forward_head(m::InceptionV3, x; pre_logits::Bool = false)
    x = m.head_drop(m.global_pool(x))
    return pre_logits ? x : m.fc(x)
end

(m::InceptionV3)(x) = forward_head(m, forward_features(m, x))

function inception_v3_stages(m::InceptionV3)
    return (Chain(m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3),
        Chain(m.Pool1, m.Conv2d_3b_1x1, m.Conv2d_4a_3x3),
        Chain(m.Pool2, m.Mixed_5b, m.Mixed_5c, m.Mixed_5d),
        Chain(m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e),
        Chain(m.Mixed_7a, m.Mixed_7b, m.Mixed_7c))
end

function forward_intermediates(m::InceptionV3, x; kwargs...)
    return sequential_intermediates(inception_v3_stages(m), x, 1:5; kwargs...)
end

function feature_info(m::InceptionV3)
    chs = (m.Conv2d_2b_3x3[:bn].chs, m.Conv2d_4a_3x3[:bn].chs,
        sum(b -> b[:bn].chs,
            (m.Mixed_5d.branch1x1, m.Mixed_5d.branch5x5_2,
                m.Mixed_5d.branch3x3dbl_3, m.Mixed_5d.branch_pool)),
        sum(b -> b[:bn].chs,
            (m.Mixed_6e.branch1x1, m.Mixed_6e.branch7x7_3,
                m.Mixed_6e.branch7x7dbl_5, m.Mixed_6e.branch_pool)),
        sum(b -> b[:bn].chs,
            (m.Mixed_7c.branch1x1, m.Mixed_7c.branch3x3_2a,
                m.Mixed_7c.branch3x3_2b, m.Mixed_7c.branch3x3dbl_3a,
                m.Mixed_7c.branch3x3dbl_3b, m.Mixed_7c.branch_pool)))
    names = ("Conv2d_2b_3x3", "Conv2d_4a_3x3", "Mixed_5d", "Mixed_6e", "Mixed_7c")
    return [(; num_chs = chs[i], reduction = 2^i, module_name = names[i]) for i in 1:5]
end

# The torchvision checkpoint also holds the auxiliary classifier, which timm drops on load.
register!(ArchSpec(TIMM, "inception_v3", inception_v3; family = "inception_v3",
    references = ["timm/inception_v3.tv_in1k" =>
        "393d84cc85c467d8fbc0dc81a65c04e87a32572c"],
    ignore = [r"^AuxLogits\."]))
