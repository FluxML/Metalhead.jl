# Ports of timm.layers shared by several families. Prefer Flux layers and NNlib functions;
# only add something here when no combination of them matches the PyTorch module.

const TRUNC_NORMAL = Flux.truncated_normal(; std = 0.02)

# Padding of a PyTorch Conv2d with `padding=k÷2`-style symmetric padding.
function get_padding(kernel_size, stride = 1, dilation = 1)
    return ((stride - 1) + dilation * (kernel_size - 1)) ÷ 2
end

# timm's to_2tuple for sizes given as a number or a (height, width) sequence.
to_2tuple(x::Number) = (x, x)
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

# The (left, right, top, bottom) TF "SAME" padding of a WHCN input, in Flux order.
function pad_same(x, kernel_size, stride, dilation)
    return ntuple(4) do i
        d = (i + 1) ÷ 2
        p = get_same_padding(size(x, d), kernel_size[d], stride[d], dilation[d])
        isodd(i) ? p ÷ 2 : p - p ÷ 2
    end
end
ChainRulesCore.@non_differentiable pad_same(::Any...)

function (m::Conv2dSame)(x)
    (; weight, stride, dilation) = m.conv
    pad = pad_same(x, size(weight), stride, dilation)
    return m.conv(NNlib.pad_zeros(x, pad; dims = (1, 2)))
end

"""
    create_conv2d(in_chs, out_chs, kernel_size; stride, dilation, padding = "", groups, depthwise, bias = false, num_experts = 0)

timm.layers.create_conv2d. Sizes are Ints or torch-order `(h, w)` pairs; a vector of kernel
sizes gives a [`MixedConv2d`](@ref) and `num_experts > 0` a [`CondConv2d`](@ref). `padding`
is `""` (PyTorch-style symmetric), `"same"` (TF-style, a `Conv2dSame` unless it is static),
`"valid"`, or an explicit torch padding.
"""
function create_conv2d(in_chs, out_chs, kernel_size; stride = 1, dilation = 1,
        padding = "", groups = 1, depthwise = false, bias = false, num_experts = 0)
    if kernel_size isa AbstractVector
        num_experts > 0 && error("MixedConv2d with CondConv is not supported")
        groups == 1 || groups == in_chs ||
            error("MixedConv2d is a normal or a depthwise conv, not $groups groups")
        return MixedConv2d(in_chs, out_chs, kernel_size; stride, dilation, padding,
            depthwise = depthwise || groups == in_chs, bias)
    end
    groups = depthwise ? in_chs : groups
    num_experts > 0 && return CondConv2d(in_chs, out_chs, kernel_size; stride, dilation,
        padding, groups, bias, num_experts)
    k, s, d = (reverse(to_2tuple(v)) for v in (kernel_size, stride, dilation))
    conv(pad) = Conv(k, in_chs => out_chs; stride = s, pad, dilation = d, groups, bias)
    padding isa AbstractString || return conv(reverse(to_2tuple(padding)))
    padding = lowercase(padding)
    padding == "valid" && return conv(0)
    is_static = all(@. s == 1 && iseven(d * (k - 1)))
    padding == "same" && !is_static && return Conv2dSame(conv(0))
    return conv(get_padding.(k, s, d))
end

# timm's _split_channels: equal splits, the first one taking the remainder.
function split_channels(channels, num_groups)
    return ntuple(num_groups) do i
        return channels ÷ num_groups + (i == 1) * (channels % num_groups)
    end
end

"""
    MixedConv2d(in_chs, out_chs, kernel_sizes; stride, dilation, padding, depthwise, bias)

timm.layers.MixedConv2d (MixConv): the channels are split into one group per kernel size,
each convolved with its own kernel size, and concatenated. A `ModuleDict` in timm, so the
convs' keys are `0.weight`, `1.weight`, ... with no segment for the container.
"""
@concrete struct MixedConv2d
    convs
    splits
end
Flux.@layer MixedConv2d

function layer_params(m::MixedConv2d)
    return [(string(i - 1, ".", name), array, kind)
            for (i, conv) in enumerate(m.convs)
            for (name, array, kind) in layer_params(conv)]
end

function MixedConv2d(in_chs, out_chs, kernel_sizes::AbstractVector; stride = 1,
        dilation = 1, padding = "", depthwise = false, bias = false)
    in_splits = split_channels(in_chs, length(kernel_sizes))
    out_splits = split_channels(out_chs, length(kernel_sizes))
    convs = map(Tuple(kernel_sizes), in_splits, out_splits) do k, in_ch, out_ch
        create_conv2d(in_ch, out_ch, k; stride, dilation, padding,
            groups = depthwise ? in_ch : 1, bias)
    end
    return MixedConv2d(convs, in_splits)
end

# Zygote can't differentiate `cumsum` of a tuple of Ints, hence a non-differentiable helper.
split_ranges(splits) = map((stop, n) -> (stop - n + 1):stop, cumsum(splits), splits)
ChainRulesCore.@non_differentiable split_ranges(::Any)

function (m::MixedConv2d)(x)
    xs = map(r -> x[:, :, r, :], split_ranges(m.splits))
    return cat(map((conv, xi) -> conv(xi), m.convs, xs)...; dims = 3)
end

"""
    CondConv2d(in_chs, out_chs, kernel_size; num_experts, stride, dilation, padding, groups, bias)

timm.layers.CondConv2d: a conv whose kernel (and bias) is a per-sample mixture of
`num_experts` kernels, called as `m(x, routing_weights)` with routing weights of size
`(num_experts, batch)`. `weight` is torch's `(num_experts, params)` reversed: column `e` is
the flattened kernel of expert `e`. The samples are convolved at once, as one grouped conv.
"""
@concrete struct CondConv2d
    weight
    bias
    kernel_size
    stride
    pad
    dilation
    groups
end
Flux.@layer CondConv2d trainable=(weight, bias)

function CondConv2d(in_chs, out_chs, kernel_size; num_experts, stride = 1, dilation = 1,
        padding = "", groups = 1, bias = false)
    k, s, d = (reverse(to_2tuple(v)) for v in (kernel_size, stride, dilation))
    lowercase(padding) in ("", "same") ||
        error("CondConv2d padding `$padding` is not supported")
    is_static = padding == "" || all(@. s == 1 && iseven(d * (k - 1)))
    # `nothing` is the dynamic TF "SAME" padding, computed from the input size
    pad = is_static ? Tuple(Iterators.flatten((p, p) for p in get_padding.(k, s, d))) :
          nothing
    # the normal(0, sqrt(2 / fan_out)) init of timm's _init_weight_goog, which all the
    # models using CondConv2d apply
    weight = randn(Float32, prod(k) * (in_chs ÷ groups) * out_chs, num_experts) .*
             sqrt(2.0f0 / (prod(k) * out_chs ÷ groups))
    return CondConv2d(weight, bias ? zeros(Float32, out_chs, num_experts) : false, k, s,
        pad, d, groups)
end

function (m::CondConv2d)(x, routing_weights)
    w, h, c, b = size(x)
    weight = reshape(m.weight * routing_weights, m.kernel_size..., c ÷ m.groups, :)
    pad = something(m.pad, pad_same(x, m.kernel_size, m.stride, m.dilation))
    y = NNlib.conv(reshape(x, w, h, c * b, 1), weight; stride = m.stride, pad,
        dilation = m.dilation, groups = m.groups * b, flipped = true)
    y = reshape(y, size(y, 1), size(y, 2), :, b)
    return m.bias isa AbstractArray ? y .+ reshape(m.bias * routing_weights, 1, 1, :, b) : y
end

# timm.layers.ConvNormAct: create_conv2d, then BatchNormAct2d, which is Flux's BatchNorm
# with the activation inside (its keys are at the BatchNorm level). With `aa_layer` and a
# stride, the conv doesn't stride and an anti-aliasing `aa` (see `create_aa`) downsamples.
function conv_norm_act(in_chs, out_chs, kernel_size = 1; stride = 1, padding = "",
        dilation = 1, groups = 1, bias = false, apply_act = true, act_layer = relu,
        eps = 1.0f-5, momentum = 0.1f0, aa_layer = nothing)
    use_aa = aa_layer !== nothing && stride > 1
    conv = create_conv2d(in_chs, out_chs, kernel_size; stride = use_aa ? 1 : stride,
        padding, dilation, groups, bias)
    bn = BatchNorm(out_chs, apply_act ? act_layer : identity; eps = Float32(eps),
        momentum = Float32(momentum))
    use_aa || return Chain(; conv, bn)
    return Chain(; conv, bn, aa = create_aa(aa_layer, out_chs; stride))
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

# DropPath on (C, L, B) tokens, or nn.Identity() when p == 0.
token_drop_path(p) = dropout_or_identity(p; dims = 3)

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

# timm.layers.SEModule (squeeze-and-excitation), in its default norm-free ReLU form.
@concrete struct SEModule
    fc1
    bn
    act
    fc2
    gate
end
Flux.@layer SEModule

function SEModule(channels::Integer; rd_ratio = 1 / 16, rd_divisor = 8)
    rd_channels = make_divisible(channels * rd_ratio, rd_divisor; round_limit = 0.0)
    return SEModule(Conv((1, 1), channels => rd_channels), identity, relu,
        Conv((1, 1), rd_channels => channels), sigmoid)
end

function (m::SEModule)(x)
    x_se = m.fc2(m.act(m.bn(m.fc1(GlobalMeanPool()(x)))))
    return x .* m.gate(x_se)
end

# timm.layers.EcaModule (efficient channel attention): a Conv1d over the channel means.
@concrete struct EcaModule
    conv
    gate
end
Flux.@layer EcaModule

function EcaModule(channels::Integer; gamma = 2, beta = 1)
    t = trunc(Int, abs(log(channels) / log(2) + beta) / gamma)  # math.log(channels, 2)
    kernel_size = max(isodd(t) ? t : t + 1, 3)
    return EcaModule(
        Conv((kernel_size,), 1 => 1; pad = (kernel_size - 1) ÷ 2, bias = false), sigmoid)
end

function (m::EcaModule)(x)
    c, b = size(x, 3), size(x, 4)
    y = m.gate(m.conv(reshape(GlobalMeanPool()(x), c, 1, b)))
    return x .* reshape(y, 1, 1, c, b)
end

# timm.layers.create_attn for the attention types "se" and "eca"; `nothing` gives no module.
function create_attn(attn_type, channels)
    attn_type === nothing && return nothing
    attn_type isa AbstractString ||
        error("`attn_layer = $(repr(attn_type))` is not supported; use \"se\" or \"eca\"")
    attn = lowercase(attn_type)
    attn == "se" && return SEModule(channels)
    attn == "eca" && return EcaModule(channels)
    return error("attention `$attn_type` is not supported; use \"se\" or \"eca\"")
end

# timm.layers.GluMlp with `gate_last = true` and no `use_conv`: fc1 gives a value half and a
# gate half along the features, `x1 * act(x2)`.
@concrete struct GluMlp
    fc1
    act
    drop1
    norm
    fc2
    drop2
end
Flux.@layer GluMlp

function GluMlp(
        in_features, hidden_features, out_features = in_features; act_layer = sigmoid,
        bias = true, drop = 0.0, init = TRUNC_NORMAL)
    iseven(hidden_features) || error("`hidden_features` ($hidden_features) must be even")
    return GluMlp(Dense(in_features => hidden_features; bias, init),
        act_layer, dropout_or_identity(drop), identity,
        Dense(hidden_features ÷ 2 => out_features; bias, init), dropout_or_identity(drop))
end

function (m::GluMlp)(x)
    x1, x2 = Flux.chunk(m.fc1(x), 2; dims = 1)
    return m.drop2(m.fc2(m.norm(m.drop1(x1 .* m.act(x2)))))
end

# timm.layers.GatedMlp: `gate_layer(hidden_features)` builds the gate (e.g. a spatial gating
# unit), which halves the features before fc2.
@concrete struct GatedMlp
    fc1
    act
    drop1
    gate
    norm
    fc2
    drop2
end
Flux.@layer GatedMlp

function GatedMlp(
        in_features, hidden_features, out_features = in_features; act_layer = gelu_erf,
        gate_layer = nothing, bias = true, drop = 0.0, init = TRUNC_NORMAL)
    gate, fc2_in = identity, hidden_features
    if gate_layer !== nothing
        iseven(hidden_features) ||
            error("`hidden_features` ($hidden_features) must be even")
        gate, fc2_in = gate_layer(hidden_features), hidden_features ÷ 2
    end
    return GatedMlp(Dense(in_features => hidden_features; bias, init), act_layer,
        dropout_or_identity(drop), gate, identity,
        Dense(fc2_in => out_features; bias, init), dropout_or_identity(drop))
end

(m::GatedMlp)(x) = m.drop2(m.fc2(m.norm(m.gate(m.drop1(m.act(m.fc1(x)))))))
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
# timm.layers.BlurPool2d: a depthwise binomial blur, then subsampling, after reflect or
# zero ("constant") padding. timm keeps the filter in a non-persistent buffer, so it is
# rebuilt here and has no keys.
@concrete struct BlurPool2d
    channels
    filt_size
    stride
    pad
    pad_mode
end
Flux.@layer BlurPool2d

function BlurPool2d(channels = nothing; filt_size = 3, stride = 2, pad_mode = "reflect")
    filt_size > 1 || error("`filt_size` must be larger than 1")
    pad_mode in ("reflect", "constant") ||
        error("`pad_mode = $(repr(pad_mode))` is not supported; use \"reflect\" or " *
              "\"constant\"")
    return BlurPool2d(channels, filt_size, stride, get_padding(filt_size, stride), pad_mode)
end

function blur_filter(T, k, c)
    coeffs = T[binomial(k - 1, i) / T(2)^(k - 1) for i in 0:(k - 1)]
    return repeat(reshape(coeffs * coeffs', k, k, 1, 1), 1, 1, 1, c)
end
ChainRulesCore.@non_differentiable blur_filter(::Any, ::Any, ::Any)

function (m::BlurPool2d)(x)
    c = something(m.channels, size(x, 3))
    filt = blur_filter(eltype(x), m.filt_size, c)
    pad = ntuple(_ -> m.pad, 4)
    x = m.pad_mode == "reflect" ? NNlib.pad_reflect(x, pad; dims = (1, 2)) :
        NNlib.pad_zeros(x, pad; dims = (1, 2))
    return NNlib.conv(x, filt; stride = m.stride, groups = c)
end

# timm.layers.create_aa for the anti-aliasing layers given by name: "avg" is
# nn.AvgPool2d(stride), "blurpool" a BlurPool2d, "blurpc" one with zero padding. Gives
# `identity` (timm's nn.Identity) when there is no aa_layer or it isn't enabled.
function create_aa(aa_layer, channels = nothing; stride = 2, enable = true)
    (aa_layer === nothing || !enable) && return identity
    aa_layer isa AbstractString ||
        error("`aa_layer = $(repr(aa_layer))` is not supported; use \"avg\", " *
              "\"blurpool\" or \"blurpc\"")
    key = replace(lowercase(aa_layer), "_" => "", "-" => "")
    key in ("avg", "avgpool") && return MeanPool((stride, stride))
    key in ("blur", "blurpool") && return BlurPool2d(channels; stride)
    key == "blurpc" && return BlurPool2d(channels; stride, pad_mode = "constant")
    return error("anti-aliasing layer `$aa_layer` is not supported; use \"avg\", " *
                 "\"blurpool\" or \"blurpc\"")
end

# timm.layers.LayerScale2d: a learned per-channel scale of a WHCN feature map.
@concrete struct LayerScale2d
    gamma
end
Flux.@layer LayerScale2d

function LayerScale2d(dim::Integer, init_values = 1e-5)
    return LayerScale2d(fill(Float32(init_values), dim))
end

(m::LayerScale2d)(x) = x .* reshape(m.gamma, 1, 1, :, 1)

"""
    MultiQueryAttention2d(dim, dim_out = dim; num_heads = 8, key_dim, value_dim,
                          kv_stride = 1, dw_kernel_size = 3, dilation = 1, padding = "",
                          attn_drop = 0.0, proj_drop = 0.0,
                          norm_layer = (; eps = 1e-5, momentum = 0.1), use_bias = false)

timm.layers.MultiQueryAttention2d on WHCN feature maps, without query strides: `num_heads`
query heads share a single key and value head, whose inputs are downsampled by a strided
depthwise conv and a BatchNorm (with the `eps` and `momentum` of `norm_layer`) when
`kv_stride > 1`.
"""
@concrete struct MultiQueryAttention2d
    query
    key
    value
    attn_drop
    output
    num_heads
end
Flux.@layer MultiQueryAttention2d

function MultiQueryAttention2d(dim, dim_out = dim; num_heads = 8, key_dim = nothing,
        value_dim = nothing, query_strides = 1, kv_stride = 1, dw_kernel_size = 3,
        dilation = 1, padding = "", attn_drop = 0.0, proj_drop = 0.0,
        norm_layer = (; eps = 1e-5, momentum = 0.1), use_bias = false)
    all(==(1), to_2tuple(query_strides)) ||
        error("`query_strides` larger than 1 are not supported by the Metalhead port of " *
              "MultiQueryAttention2d")
    key_dim = something(key_dim, dim ÷ num_heads)
    value_dim = something(value_dim, dim ÷ num_heads)
    function kv_branch(out_dim; proj_padding = "")
        proj = create_conv2d(dim, out_dim, 1; padding = proj_padding, bias = use_bias)
        kv_stride > 1 || return Chain(; proj)
        down_conv = create_conv2d(dim, dim, dw_kernel_size; stride = kv_stride, dilation,
            padding, depthwise = true)
        norm = BatchNorm(dim; eps = Float32(norm_layer.eps),
            momentum = Float32(norm_layer.momentum))
        return Chain(; down_conv, norm, proj)
    end
    query = Chain(; proj = create_conv2d(dim, num_heads * key_dim, 1; bias = use_bias))
    output = Chain(;
        proj = create_conv2d(value_dim * num_heads, dim_out, 1; bias = use_bias),
        drop = dropout_or_identity(proj_drop))
    return MultiQueryAttention2d(query, kv_branch(key_dim; proj_padding = padding),
        kv_branch(value_dim), dropout_or_identity(attn_drop), output, num_heads)
end

function (m::MultiQueryAttention2d)(x)
    q, k, v = m.query(x), m.key(x), m.value(x)
    w, h, _, b = size(q)
    key_dim, value_dim = size(k, 3), size(v, 3)
    # torch's (B, heads, key_dim, H * W) query and (B, 1, H' * W', dim) keys and values,
    # in NNlib's (dim, heads, length, batch) layout
    q = permutedims(reshape(q, w * h, key_dim, m.num_heads, b), (2, 3, 1, 4))
    k = permutedims(reshape(k, :, key_dim, 1, b), (2, 3, 1, 4))
    v = permutedims(reshape(v, :, value_dim, 1, b), (2, 3, 1, 4))
    o = NNlib.scaled_dot_product_attention(q, k, v; fdrop = m.attn_drop)
    o = reshape(permutedims(o, (3, 1, 2, 4)), w, h, value_dim * m.num_heads, b)
    return m.output(o)
end

# timm.layers.get_act_layer for the activations given by name (a function passes through).
const ACT_LAYERS = Dict(
    "relu" => relu, "relu6" => relu6, "silu" => swish, "swish" => swish, "mish" => mish,
    "hard_sigmoid" => hardsigmoid, "hard_swish" => hardswish, "sigmoid" => sigmoid,
    "gelu" => gelu_erf)

get_act_layer(name::Nothing) = nothing
get_act_layer(f::Function) = f
function get_act_layer(name::AbstractString)
    return get(ACT_LAYERS, lowercase(name)) do
        error("activation `$name` is not supported by the Metalhead ports; " *
              "use one of $(join(sort!(collect(keys(ACT_LAYERS))), ", "))")
    end
end
