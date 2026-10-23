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
