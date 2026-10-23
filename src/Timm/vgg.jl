# Ported from timm.models.vgg (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/vgg_* and test/references/timm--vgg*.

const VGG_CFGS = Dict(
    "vgg11" => (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "vgg13" => (64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "vgg16" => (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M"),
    "vgg19" => (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512, "M"))

# Conv2d init of torch's `_initialize_weights`: kaiming_normal_(fan_out).
function kaiming_normal_fan_out(dims...)
    return randn(Float32, dims...) .* sqrt(2.0f0 / (prod(dims[1:(end - 2)]) * dims[end]))
end

# torch's adaptive_avg_pool2d to `outsize` = (W, H): output i averages the input range
# [floor(i*n_in/n_out), ceil((i+1)*n_in/n_out)), as a matrix per dimension.
function adaptive_pool_matrix(T, n_in, n_out)
    window(i) = (((i - 1) * n_in) ÷ n_out + 1):cld(i * n_in, n_out)
    return [j in window(i) ? inv(T(length(window(i)))) : zero(T)
            for i in 1:n_out, j in 1:n_in]
end
ChainRulesCore.@non_differentiable adaptive_pool_matrix(::Any, ::Any, ::Any)

function adaptive_avg_pool2d(x, outsize)
    w, h, c, b = size(x)
    if outsize[1] != w
        x = reshape(adaptive_pool_matrix(eltype(x), w, outsize[1]) * reshape(x, w, :),
            outsize[1], h, c, b)
    end
    if outsize[2] != h
        P = adaptive_pool_matrix(eltype(x), h, outsize[2])
        x = permutedims(x, (2, 1, 3, 4))
        x = reshape(P * reshape(x, h, :), outsize[2], outsize[1], c, b)
        x = permutedims(x, (2, 1, 3, 4))
    end
    return x
end

# timm.models.vgg.ConvMlp: the VGG head as convolutions, which keeps the input >= 7x7.
@concrete struct VGGConvMlp
    fc1
    act1
    drop
    fc2
    act2
    input_kernel_size
end
Flux.@layer VGGConvMlp

function VGGConvMlp(in_features = 512, out_features = 4096, kernel_size = 7;
        mlp_ratio = 1.0, drop_rate = 0.2)
    mid_features = trunc(Int, out_features * mlp_ratio)
    return VGGConvMlp(
        Conv((kernel_size, kernel_size), in_features => mid_features;
            init = kaiming_normal_fan_out),
        relu, dropout_or_identity(drop_rate),
        Conv((1, 1), mid_features => out_features; init = kaiming_normal_fan_out),
        relu, kernel_size)
end

function (m::VGGConvMlp)(x)
    k = m.input_kernel_size
    if size(x, 1) < k || size(x, 2) < k
        x = adaptive_avg_pool2d(x, (max(k, size(x, 1)), max(k, size(x, 2))))
    end
    return m.act2(m.fc2(m.drop(m.act1(m.fc1(x)))))
end

@concrete struct VGG
    features
    pre_logits
    head
end
Flux.@layer VGG

"""
    vgg(; cfg, num_classes = 1000, in_chans = 3, norm_layer = nothing, kwargs...)

Port of timm's `VGG`. `cfg` lists the output channels of the 3x3 convolutions, with `"M"`
for a max pool, and `norm_layer = "batchnorm2d"` adds batch normalisation (the `_bn`
variants). Other keyword arguments are those of the timm class.
"""
function vgg(; cfg, num_classes = 1000, in_chans = 3, output_stride = 32, mlp_ratio = 1.0,
        act_layer = "relu", conv_layer = "conv2d", norm_layer = nothing,
        global_pool = "avg", drop_rate = 0.0)
    check_unsupported(
        "VGG"; output_stride = (output_stride, 32), act_layer = (act_layer, "relu"),
        conv_layer = (conv_layer, "conv2d"), global_pool = (global_pool, "avg"))
    norm_layer in (nothing, "batchnorm2d") ||
        error("`norm_layer = $(repr(norm_layer))` is not supported by the Metalhead port " *
              "of VGG; use \"batchnorm2d\" or nothing")
    layers = Any[]
    prev_chs = in_chans
    for v in cfg
        if v == "M"
            push!(layers, MaxPool((2, 2); stride = 2))
        else
            push!(layers,
                Conv((3, 3), prev_chs => v; pad = 1, init = kaiming_normal_fan_out))
            norm_layer === nothing || push!(layers, BatchNorm(v))
            push!(layers, relu)
            prev_chs = v
        end
    end
    head_hidden_size = 4096
    return VGG(Chain(layers),
        VGGConvMlp(prev_chs, head_hidden_size, 7; mlp_ratio, drop_rate),
        ClassifierHead(head_hidden_size, num_classes; pool_type = global_pool, drop_rate))
end

forward_features(m::VGG, x) = m.features(x)
function forward_head(m::VGG, x; pre_logits::Bool = false)
    return m.head(m.pre_logits(x); pre_logits)
end
(m::VGG)(x) = forward_head(m, forward_features(m, x))

const VGG_REFERENCES = Dict(
    "vgg11" => ["timm/vgg11.tv_in1k" => "de67e844ca0254353694aba9167f0aac078cc29c"],
    "vgg11_bn" => ["timm/vgg11_bn.tv_in1k" => "d1b4a6d788dbf60b7dd034952e34dfda1d52404d"]
)

for depth in (11, 13, 16, 19), bn in (false, true)
    name = string("vgg", depth, bn ? "_bn" : "")
    defaults = (; cfg = VGG_CFGS["vgg$depth"],
        (bn ? (; norm_layer = "batchnorm2d") : (;))...)
    register!(ArchSpec(TIMM, name, vgg; defaults,
        references = get(VGG_REFERENCES, name, [])))
end
