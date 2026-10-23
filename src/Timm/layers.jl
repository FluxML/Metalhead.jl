# Ports of timm.layers shared by several families. Prefer Flux layers and NNlib functions;
# only add something here when no combination of them matches the PyTorch module.

const TRUNC_NORMAL = Flux.truncated_normal(; std = 0.02)

# Padding of a PyTorch Conv2d with `padding=k÷2`-style symmetric padding.
function get_padding(kernel_size, stride = 1, dilation = 1)
    return ((stride - 1) + dilation * (kernel_size - 1)) ÷ 2
end

# timm's to_2tuple for sizes given as an Int or a (height, width) sequence.
to_2tuple(x::Integer) = (x, x)
to_2tuple(x) = Tuple(x)

# timm.layers.make_divisible: round a channel count to a multiple of `divisor`.
function make_divisible(v, divisor = 8; min_value = nothing, round_limit = 0.9)
    min_value = isnothing(min_value) || iszero(min_value) ? divisor : min_value
    new_v = max(min_value, trunc(Int, v + divisor / 2) ÷ divisor * divisor)
    return new_v < round_limit * v ? new_v + divisor : new_v
end

# TF "SAME" padding of one spatial dimension of size n (timm.layers.get_same_padding).
get_same_padding(n, k, s, d) = max((cld(n, s) - 1) * s + (k - 1) * d + 1 - n, 0)

# timm.layers.Conv2dSame: a Conv with TF "SAME" padding computed from the input size, the
# extra pixel going last. It subclasses nn.Conv2d, so its keys are those of the wrapped Conv.
@concrete struct Conv2dSame
    conv
end
Flux.@layer Conv2dSame
layer_params(m::Conv2dSame) = layer_params(m.conv)

function (m::Conv2dSame)(x)
    (; weight, stride, dilation) = m.conv
    pad = ntuple(4) do i
        d = (i + 1) ÷ 2
        p = get_same_padding(size(x, d), size(weight, d), stride[d], dilation[d])
        isodd(i) ? p ÷ 2 : p - p ÷ 2
    end
    return m.conv(NNlib.pad_zeros(x, pad; dims = (1, 2)))
end

"""
    create_conv2d(in_chs, out_chs, kernel_size; stride, dilation, padding = "", groups, depthwise, bias = false)

timm.layers.create_conv2d for one kernel size (no MixedConv2d/CondConv2d). Sizes are Ints or
torch-order `(h, w)` pairs. `padding` is `""` (PyTorch-style symmetric), `"same"` (TF-style,
a [`Conv2dSame`](@ref) unless it is static), `"valid"`, or an explicit torch padding.
"""
function create_conv2d(in_chs, out_chs, kernel_size; stride = 1, dilation = 1,
        padding = "", groups = 1, depthwise = false, bias = false)
    k, s, d = (reverse(to_2tuple(v)) for v in (kernel_size, stride, dilation))
    conv(pad) = Conv(k, in_chs => out_chs; stride = s, pad, dilation = d,
        groups = depthwise ? in_chs : groups, bias)
    padding isa AbstractString || return conv(reverse(to_2tuple(padding)))
    padding = lowercase(padding)
    padding == "valid" && return conv(0)
    is_static = all(@. s == 1 && iseven(d * (k - 1)))
    padding == "same" && !is_static && return Conv2dSame(conv(0))
    return conv(get_padding.(k, s, d))
end

# timm.layers.ConvNormAct without anti-aliasing: create_conv2d, then BatchNormAct2d, which
# is Flux's BatchNorm with the activation inside (its keys are at the BatchNorm level).
function conv_norm_act(in_chs, out_chs, kernel_size = 1; stride = 1, padding = "",
        dilation = 1, groups = 1, bias = false, apply_act = true, act_layer = relu,
        eps = 1.0f-5)
    return Chain(;
        conv = create_conv2d(
            in_chs, out_chs, kernel_size; stride, padding, dilation, groups, bias),
        bn = BatchNorm(out_chs, apply_act ? act_layer : identity; eps = Float32(eps)))
end

# nn.AvgPool2d(k, stride, ceil_mode=True, count_include_pad=False)
function avg_pool_ceil(x, k::Int, stride::Int)
    pad(n) = max(0, cld(n - k, stride) * stride + k - n)
    return NNlib.meanpool(
        x, (k, k); stride, pad = (0, pad(size(x, 1)), 0, pad(size(x, 2))),
        count_include_pad = false)
end

# timm's SelectAdaptivePool2d(pool_type="avg")
function select_adaptive_pool2d(; flatten::Bool)
    return Chain(; pool = GlobalMeanPool(), flatten = flatten ? Flux.flatten : identity)
end

# timm.layers.create_classifier with a Linear classifier: (global_pool, fc), or
# (global_pool, dropout, fc) when `drop_rate` is given.
function create_classifier(num_features, num_classes; pool_type = "avg", drop_rate = nothing)
    pool_type == "avg" || error("global pool `$pool_type` is not supported")
    global_pool = select_adaptive_pool2d(; flatten = true)
    fc = num_classes > 0 ? Dense(num_features => num_classes) : identity
    return drop_rate === nothing ? (global_pool, fc) : (global_pool, Dropout(drop_rate), fc)
end

# timm's DropPath (stochastic depth) drops whole samples: `dims` is the batch dimension
# (4 for WHCN feature maps, 3 for (C, L, B) tokens). For modules timm creates only when p > 0.
drop_path(p; dims = 4) = p > 0 ? Dropout(p; dims) : nothing

# nn.Dropout(p) / DropPath(p), or nn.Identity() when p == 0.
dropout_or_identity(p; dims = :) = p > 0 ? Dropout(p; dims) : identity

function check_unsupported(name; kwargs...)
    for (k, (v, default)) in pairs(kwargs)
        v == default ||
            error("`$k = $(repr(v))` is not supported by the Metalhead port of $name")
    end
end

# timm.layers.LayerNorm2d: LayerNorm over the channels of a WHCN feature map.
@concrete struct LayerNorm2d
    weight
    bias
    eps
end
Flux.@layer LayerNorm2d

function LayerNorm2d(channels::Integer; eps = 1e-6)
    return LayerNorm2d(ones(Float32, channels), zeros(Float32, channels), eps)
end

function (m::LayerNorm2d)(x)
    y = NNlib.normalise(x; dims = 3, eps = m.eps)
    return y .* reshape(m.weight, 1, 1, :, 1) .+ reshape(m.bias, 1, 1, :, 1)
end

# timm.layers.Mlp, with 1x1 convolutions on WHCN when `use_conv = true`.
@concrete struct Mlp
    fc1
    act
    drop1
    norm
    fc2
    drop2
end
Flux.@layer Mlp

function Mlp(
        in_features, hidden_features, out_features = in_features; act_layer = gelu_erf,
        bias = true, drop = 0.0, use_conv = false, init = TRUNC_NORMAL)
    linear(i, o) = use_conv ? Conv((1, 1), i => o; bias, init) : Dense(i => o; bias, init)
    return Mlp(linear(in_features, hidden_features),
        act_layer, dropout_or_identity(drop), identity,
        linear(hidden_features, out_features), dropout_or_identity(drop))
end

(m::Mlp)(x) = m.drop2(m.fc2(m.norm(m.drop1(m.act(m.fc1(x))))))

# timm.layers.PatchEmbed: (W, H, C, B) image to (C, W*H, B) tokens.
@concrete struct PatchEmbed
    proj
end
Flux.@layer PatchEmbed

function (m::PatchEmbed)(x)
    x = m.proj(x)
    w, h, c, b = size(x)
    return permutedims(reshape(x, w * h, c, b), (2, 1, 3))
end

# timm.layers.NormMlpClassifierHead without the hidden layer: pool, norm, flatten, dropout, fc.
@concrete struct NormMlpClassifierHead
    global_pool
    norm
    flatten
    pre_logits
    drop
    fc
end
Flux.@layer NormMlpClassifierHead

function NormMlpClassifierHead(in_features, num_classes; drop_rate = 0.0)
    return NormMlpClassifierHead(
        select_adaptive_pool2d(; flatten = false), LayerNorm2d(in_features),
        Flux.flatten, identity, dropout_or_identity(drop_rate),
        num_classes > 0 ? Dense(in_features => num_classes; init = TRUNC_NORMAL) :
        identity)
end

function (m::NormMlpClassifierHead)(x; pre_logits::Bool = false)
    x = m.drop(m.pre_logits(m.flatten(m.norm(m.global_pool(x)))))
    return pre_logits ? x : m.fc(x)
end

# timm.layers.ClassifierHead (avg pool, dropout, Linear) of the `use_conv = false` kind.
@concrete struct ClassifierHead
    global_pool
    drop
    fc
    flatten
end
Flux.@layer ClassifierHead

function ClassifierHead(in_features, num_classes; pool_type = "avg", drop_rate = 0.0)
    global_pool, drop, fc = create_classifier(in_features, num_classes; pool_type, drop_rate)
    return ClassifierHead(global_pool, drop, fc, identity)
end

function (m::ClassifierHead)(x; pre_logits::Bool = false)
    x = m.drop(m.global_pool(x))
    return pre_logits ? m.flatten(x) : m.flatten(m.fc(x))
end
