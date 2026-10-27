# Ported from timm.models._efficientnet_blocks (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/efficientnet_* and test/references/timm--*efficientnet*.
#
# Not ported: MobileAttention with multi-head attention (`mha` blocks, which no timm
# architecture uses) or a CPE, anti-aliasing in the DS and IR blocks, space-to-depth,
# and norm layers other than BatchNorm2d.

# The arguments of timm's `partial(nn.BatchNorm2d, eps = ..., momentum = ...)` norm_layer.
const BATCHNORM2D = (; eps = 1e-5, momentum = 0.1)

# timm's BatchNormAct2d, whose parameters are those of the BatchNorm:
# `get_norm_act_layer(norm_layer, act_layer)(channels, apply_act = apply_act)`.
function norm_act(norm_layer, channels, act_layer; apply_act = true)
    return init_layer(BatchNorm, channels, apply_act ? act_layer : identity;
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
    return SqueezeExcite(init_layer(Conv, (1, 1), in_chs => rd_channels), act_layer,
        init_layer(Conv, (1, 1), rd_channels => in_chs), gate_layer)
end

function (m::SqueezeExcite)(x)
    x_se = mean(x; dims = (1, 2))
    x_se = m.conv_expand(m.act1(m.conv_reduce(x_se)))
    return x .* m.gate(x_se)
end

@concrete struct ConvBnAct
    conv
    bn1
    aa
    drop_path
    has_skip
end
Flux.@layer ConvBnAct

function ConvBnAct(in_chs, out_chs; kernel_size, stride = 1, dilation = 1, group_size = 0,
        pad_type = "", skip = false, act_layer = relu, norm_layer = BATCHNORM2D,
        aa_layer = nothing, drop_path_rate = 0.0)
    use_aa = aa_layer !== nothing && stride > 1
    conv = create_conv2d(in_chs, out_chs, kernel_size; stride = use_aa ? 1 : stride,
        dilation, groups = num_groups(group_size, in_chs), padding = pad_type)
    return ConvBnAct(conv, norm_act(norm_layer, out_chs, act_layer),
        create_aa(aa_layer, out_chs; stride, enable = use_aa),
        dropout_or_identity(drop_path_rate; dims = 4),
        skip && stride == 1 && in_chs == out_chs)
end

function (m::ConvBnAct)(x)
    shortcut = x
    x = m.aa(m.bn1(m.conv(x)))
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

# Universal inverted residual (UIB) of MobileNetV4: an IR block whose depthwise convs before
# the expansion (`dw_start`), after it (`dw_mid`) and after the projection (`dw_end`) are
# each optional (kernel size 0), followed by an optional layer scale.
@concrete struct UniversalInvertedResidual
    dw_start
    pw_exp
    dw_mid
    se
    pw_proj
    dw_end
    layer_scale
    drop_path
    has_skip
end
Flux.@layer UniversalInvertedResidual

function UniversalInvertedResidual(in_chs, out_chs; dw_kernel_size_start = 0,
        dw_kernel_size_mid = 3, dw_kernel_size_end = 0, stride = 1, dilation = 1,
        group_size = 1, pad_type = "", noskip = false, exp_ratio = 1.0, act_layer = relu,
        norm_layer = BATCHNORM2D, aa_layer = nothing, se_layer = nothing,
        drop_path_rate = 0.0, layer_scale_init_value = 1e-5)
    stride > 1 && dw_kernel_size_start == dw_kernel_size_mid == dw_kernel_size_end == 0 &&
        error("a strided UniversalInvertedResidual needs a depthwise conv")
    cna(i, o, k; kw...) = conv_norm_act(i, o, k; padding = pad_type, act_layer,
        eps = norm_layer.eps, momentum = norm_layer.momentum, kw...)
    dw_start = iszero(dw_kernel_size_start) ? identity :
               cna(in_chs, in_chs, dw_kernel_size_start;
        stride = iszero(dw_kernel_size_mid) ? stride : 1, dilation,
        groups = num_groups(group_size, in_chs), apply_act = false, aa_layer)
    mid_chs = make_divisible(in_chs * exp_ratio)
    dw_mid = iszero(dw_kernel_size_mid) ? identity :
             cna(mid_chs, mid_chs, dw_kernel_size_mid; stride, dilation,
        groups = num_groups(group_size, mid_chs), aa_layer)
    dw_end = identity
    if !iszero(dw_kernel_size_end)
        dw_end_stride = dw_kernel_size_start == dw_kernel_size_mid == 0 ? stride : 1
        dw_end_stride > 1 && aa_layer !== nothing &&
            error("anti-aliasing is not supported in a strided `dw_end`")
        dw_end = cna(out_chs, out_chs, dw_kernel_size_end; stride = dw_end_stride,
            dilation, groups = num_groups(group_size, out_chs), apply_act = false)
    end
    return UniversalInvertedResidual(dw_start, cna(in_chs, mid_chs, 1), dw_mid,
        se_layer === nothing ? identity : se_layer(mid_chs; act_layer),
        cna(mid_chs, out_chs, 1; apply_act = false), dw_end,
        layer_scale_init_value === nothing ? identity :
        LayerScale2d(out_chs, layer_scale_init_value),
        dropout_or_identity(drop_path_rate; dims = 4),
        in_chs == out_chs && stride == 1 && !noskip)
end

function (m::UniversalInvertedResidual)(x)
    shortcut = x
    x = m.dw_mid(m.pw_exp(m.dw_start(x)))
    x = m.layer_scale(m.dw_end(m.pw_proj(m.se(x))))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end

# Mobile attention block of MobileNetV4 (`mqa` blocks): a BatchNorm, multi-query attention
# and an optional layer scale, in a residual.
@concrete struct MobileAttention
    norm
    attn
    layer_scale
    drop_path
    has_skip
end
Flux.@layer MobileAttention

function MobileAttention(in_chs, out_chs; stride = 1, dw_kernel_size = 3, dilation = 1,
        group_size = 1, pad_type = "", num_heads = 8, key_dim = 64, value_dim = 64,
        use_multi_query = false, query_strides = 1, kv_stride = 1, noskip = false,
        act_layer = relu, norm_layer = BATCHNORM2D, aa_layer = nothing,
        drop_path_rate = 0.0, attn_drop = 0.0, proj_drop = 0.0,
        layer_scale_init_value = 1e-5, use_bias = false, use_cpe = false)
    use_multi_query ||
        error("MobileAttention with multi-head attention (`mha` blocks) is not supported " *
              "by the Metalhead ports of the EfficientNet family")
    check_unsupported("MobileAttention"; use_cpe = (use_cpe, false))
    num_heads = something(num_heads, in_chs ÷ key_dim)
    attn = MultiQueryAttention2d(in_chs, out_chs; num_heads, key_dim, value_dim,
        query_strides, kv_stride, dw_kernel_size, dilation, padding = pad_type, attn_drop,
        proj_drop, norm_layer)
    return MobileAttention(norm_act(norm_layer, in_chs, act_layer; apply_act = false), attn,
        layer_scale_init_value === nothing ? identity :
        LayerScale2d(out_chs, layer_scale_init_value),
        dropout_or_identity(drop_path_rate; dims = 4),
        stride == 1 && in_chs == out_chs && !noskip)
end

function (m::MobileAttention)(x)
    shortcut = x
    x = m.layer_scale(m.attn(m.norm(x)))
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
        dropout_or_identity(drop_path_rate; dims = 4),
        init_layer(Dense, in_chs => num_experts),
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
    aa
    se
    conv_pwl
    bn2
    drop_path
    has_skip
end
Flux.@layer EdgeResidual

function EdgeResidual(in_chs, out_chs; exp_kernel_size = 3, stride = 1, dilation = 1,
        group_size = 0, pad_type = "", force_in_chs = 0, noskip = false, exp_ratio = 1.0,
        pw_kernel_size = 1, act_layer = relu, norm_layer = BATCHNORM2D, aa_layer = nothing,
        se_layer = nothing, drop_path_rate = 0.0)
    mid_chs = make_divisible((force_in_chs > 0 ? force_in_chs : in_chs) * exp_ratio)
    use_aa = aa_layer !== nothing && stride > 1
    conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size;
        stride = use_aa ? 1 : stride, dilation, groups = num_groups(group_size, mid_chs),
        padding = pad_type)
    se = se_layer === nothing ? identity : se_layer(mid_chs; act_layer)
    return EdgeResidual(
        conv_exp, norm_act(norm_layer, mid_chs, act_layer),
        create_aa(aa_layer, mid_chs; stride, enable = use_aa), se,
        create_conv2d(mid_chs, out_chs, pw_kernel_size; padding = pad_type),
        norm_act(norm_layer, out_chs, act_layer; apply_act = false),
        dropout_or_identity(drop_path_rate; dims = 4),
        in_chs == out_chs && stride == 1 && !noskip)
end

function (m::EdgeResidual)(x)
    shortcut = x
    x = m.aa(m.bn1(m.conv_exp(x)))
    x = m.bn2(m.conv_pwl(m.se(x)))
    return m.has_skip ? m.drop_path(x) .+ shortcut : x
end
