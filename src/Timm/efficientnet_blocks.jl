# Ported from timm.models._efficientnet_blocks (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/efficientnet_* and test/references/timm--*efficientnet*.
#
# Not ported: UniversalInvertedResidual, MobileAttention, anti-aliasing, space-to-depth,
# and norm layers other than BatchNorm2d.

# The arguments of timm's `partial(nn.BatchNorm2d, eps = ..., momentum = ...)` norm_layer.
const BATCHNORM2D = (; eps = 1e-5, momentum = 0.1)

# timm's BatchNormAct2d, whose parameters are those of the BatchNorm:
# `get_norm_act_layer(norm_layer, act_layer)(channels, apply_act = apply_act)`.
function norm_act(norm_layer, channels, act_layer; apply_act = true)
    return BatchNorm(channels, apply_act ? act_layer : identity;
        eps = Float32(norm_layer.eps), momentum = Float32(norm_layer.momentum))
end

# Number of groups of a conv with `group_size` channels per group (0: a normal conv).
function num_groups(group_size, channels)
    (group_size === nothing || group_size == 0) && return 1
    channels % group_size == 0 ||
        error("$channels channels are not divisible into groups of $group_size")
    return channels ÷ group_size
end

# Squeeze-and-Excitation with the options of the EfficientNet/MobileNet family.
@concrete struct SqueezeExcite
    conv_reduce
    act1
    conv_expand
    gate
end
Flux.@layer SqueezeExcite

function SqueezeExcite(in_chs; rd_ratio = 0.25, rd_channels = nothing, act_layer = relu,
        gate_layer = sigmoid, force_act_layer = nothing, rd_round_fn = nothing)
    if rd_channels === nothing
        rd_round_fn = something(rd_round_fn, x -> round(Int, x))
        rd_channels = rd_round_fn(in_chs * rd_ratio)
    end
    act_layer = something(force_act_layer, act_layer)
    return SqueezeExcite(Conv((1, 1), in_chs => rd_channels), act_layer,
        Conv((1, 1), rd_channels => in_chs), gate_layer)
end

function (m::SqueezeExcite)(x)
    x_se = mean(x; dims = (1, 2))
    x_se = m.conv_expand(m.act1(m.conv_reduce(x_se)))
    return x .* m.gate(x_se)
end

@concrete struct ConvBnAct
    conv
    bn1
    drop_path
    has_skip
end
Flux.@layer ConvBnAct

function ConvBnAct(in_chs, out_chs; kernel_size, stride = 1, dilation = 1, group_size = 0,
        pad_type = "", skip = false, act_layer = relu, norm_layer = BATCHNORM2D,
        drop_path_rate = 0.0)
    conv = create_conv2d(in_chs, out_chs, kernel_size; stride, dilation,
        groups = num_groups(group_size, in_chs), padding = pad_type)
    return ConvBnAct(conv, norm_act(norm_layer, out_chs, act_layer),
        dropout_or_identity(drop_path_rate; dims = 4),
        skip && stride == 1 && in_chs == out_chs)
end

function (m::ConvBnAct)(x)
    shortcut = x
    x = m.bn1(m.conv(x))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end

# Depthwise-separable block, also the stand-in for IR blocks without expansion.
@concrete struct DepthwiseSeparableConv
    conv_dw
    bn1
    se
    conv_pw
    bn2
    drop_path
    has_skip
end
Flux.@layer DepthwiseSeparableConv

function DepthwiseSeparableConv(in_chs, out_chs; dw_kernel_size = 3, stride = 1,
        dilation = 1, group_size = 1, pad_type = "", noskip = false, pw_kernel_size = 1,
        pw_act = false, act_layer = relu, norm_layer = BATCHNORM2D, se_layer = nothing,
        drop_path_rate = 0.0)
    conv_dw = create_conv2d(in_chs, in_chs, dw_kernel_size; stride, dilation,
        padding = pad_type, groups = num_groups(group_size, in_chs))
    se = se_layer === nothing ? identity : se_layer(in_chs; act_layer)
    return DepthwiseSeparableConv(
        conv_dw, norm_act(norm_layer, in_chs, act_layer), se,
        create_conv2d(in_chs, out_chs, pw_kernel_size; padding = pad_type),
        norm_act(norm_layer, out_chs, act_layer; apply_act = pw_act),
        dropout_or_identity(drop_path_rate; dims = 4),
        stride == 1 && in_chs == out_chs && !noskip)
end

function (m::DepthwiseSeparableConv)(x)
    shortcut = x
    x = m.bn1(m.conv_dw(x))
    x = m.bn2(m.conv_pw(m.se(x)))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end

# Inverted residual block (MBConv) with optional SE.
@concrete struct InvertedResidual
    conv_pw
    bn1
    conv_dw
    bn2
    se
    conv_pwl
    bn3
    drop_path
    has_skip
end
Flux.@layer InvertedResidual

function InvertedResidual(in_chs, out_chs; dw_kernel_size = 3, stride = 1, dilation = 1,
        group_size = 1, pad_type = "", noskip = false, exp_ratio = 1.0, exp_kernel_size = 1,
        pw_kernel_size = 1, act_layer = relu, norm_layer = BATCHNORM2D, se_layer = nothing,
        drop_path_rate = 0.0)
    mid_chs = make_divisible(in_chs * exp_ratio)
    conv_dw = create_conv2d(mid_chs, mid_chs, dw_kernel_size; stride, dilation,
        padding = pad_type, groups = num_groups(group_size, mid_chs))
    se = se_layer === nothing ? identity : se_layer(mid_chs; act_layer)
    return InvertedResidual(
        create_conv2d(in_chs, mid_chs, exp_kernel_size; padding = pad_type),
        norm_act(norm_layer, mid_chs, act_layer), conv_dw,
        norm_act(norm_layer, mid_chs, act_layer), se,
        create_conv2d(mid_chs, out_chs, pw_kernel_size; padding = pad_type),
        norm_act(norm_layer, out_chs, act_layer; apply_act = false),
        dropout_or_identity(drop_path_rate; dims = 4),
        in_chs == out_chs && stride == 1 && !noskip)
end

function (m::InvertedResidual)(x)
    shortcut = x
    x = m.bn1(m.conv_pw(x))
    x = m.bn2(m.conv_dw(x))
    x = m.bn3(m.conv_pwl(m.se(x)))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end

# Inverted residual block whose convs are CondConv2d, mixed per sample by routing weights
# computed from the pooled input.
@concrete struct CondConvResidual
    conv_pw
    bn1
    conv_dw
    bn2
    se
    conv_pwl
    bn3
    drop_path
    routing_fn
    has_skip
end
Flux.@layer CondConvResidual

function CondConvResidual(in_chs, out_chs; dw_kernel_size = 3, stride = 1, dilation = 1,
        group_size = 1, pad_type = "", noskip = false, exp_ratio = 1.0, exp_kernel_size = 1,
        pw_kernel_size = 1, act_layer = relu, norm_layer = BATCHNORM2D, se_layer = nothing,
        num_experts = 0, drop_path_rate = 0.0)
    mid_chs = make_divisible(in_chs * exp_ratio)
    conv_dw = create_conv2d(mid_chs, mid_chs, dw_kernel_size; stride, dilation,
        padding = pad_type, groups = num_groups(group_size, mid_chs), num_experts)
    se = se_layer === nothing ? identity : se_layer(mid_chs; act_layer)
    return CondConvResidual(
        create_conv2d(in_chs, mid_chs, exp_kernel_size; padding = pad_type, num_experts),
        norm_act(norm_layer, mid_chs, act_layer), conv_dw,
        norm_act(norm_layer, mid_chs, act_layer), se,
        create_conv2d(mid_chs, out_chs, pw_kernel_size; padding = pad_type, num_experts),
        norm_act(norm_layer, out_chs, act_layer; apply_act = false),
        dropout_or_identity(drop_path_rate; dims = 4), Dense(in_chs => num_experts),
        in_chs == out_chs && stride == 1 && !noskip)
end

function (m::CondConvResidual)(x)
    shortcut = x
    routing_weights = sigmoid.(m.routing_fn(reshape(mean(x; dims = (1, 2)), size(x, 3), :)))
    x = m.bn1(m.conv_pw(x, routing_weights))
    x = m.bn2(m.conv_dw(x, routing_weights))
    x = m.bn3(m.conv_pwl(m.se(x), routing_weights))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end

# Residual block with an expansion conv and a pointwise-linear projection (FusedMBConv).
@concrete struct EdgeResidual
    conv_exp
    bn1
    se
    conv_pwl
    bn2
    drop_path
    has_skip
end
Flux.@layer EdgeResidual

function EdgeResidual(in_chs, out_chs; exp_kernel_size = 3, stride = 1, dilation = 1,
        group_size = 0, pad_type = "", force_in_chs = 0, noskip = false, exp_ratio = 1.0,
        pw_kernel_size = 1, act_layer = relu, norm_layer = BATCHNORM2D, se_layer = nothing,
        drop_path_rate = 0.0)
    mid_chs = make_divisible((force_in_chs > 0 ? force_in_chs : in_chs) * exp_ratio)
    conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size; stride, dilation,
        groups = num_groups(group_size, mid_chs), padding = pad_type)
    se = se_layer === nothing ? identity : se_layer(mid_chs; act_layer)
    return EdgeResidual(
        conv_exp, norm_act(norm_layer, mid_chs, act_layer), se,
        create_conv2d(mid_chs, out_chs, pw_kernel_size; padding = pad_type),
        norm_act(norm_layer, out_chs, act_layer; apply_act = false),
        dropout_or_identity(drop_path_rate; dims = 4),
        in_chs == out_chs && stride == 1 && !noskip)
end

function (m::EdgeResidual)(x)
    shortcut = x
    x = m.bn1(m.conv_exp(x))
    x = m.bn2(m.conv_pwl(m.se(x)))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end
