# Ported from timm.models.densenet (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/densenet_* and test/references/timm--densenet*.

# timm's DenseLayer on the concatenation of all previous features; returns the new features.
@concrete struct DenseLayer
    norm1
    conv1
    norm2
    conv2
    drop
end
Flux.@layer DenseLayer

function DenseLayer(num_input_features, growth_rate, bn_size; drop_rate = 0.0)
    return DenseLayer(
        BatchNorm(num_input_features, relu),
        Conv((1, 1), num_input_features => bn_size * growth_rate; bias = false,
            init = Flux.kaiming_normal),
        BatchNorm(bn_size * growth_rate, relu),
        Conv((3, 3), bn_size * growth_rate => growth_rate; pad = 1, bias = false,
            init = Flux.kaiming_normal),
        dropout_or_identity(drop_rate))
end

(m::DenseLayer)(x) = m.drop(m.conv2(m.norm2(m.conv1(m.norm1(x)))))

concat_features(new_features, features) = cat(features, new_features; dims = 3)

# timm's DenseBlock (a ModuleDict of dense layers) feeds each layer the concatenation of the
# block input and all earlier outputs, which is what a skip connection with `cat` does.
function dense_block(num_layers, num_input_features, bn_size, growth_rate; drop_rate = 0.0)
    layers = [Symbol(:denselayer, i) => SkipConnection(
                  DenseLayer(num_input_features + (i - 1) * growth_rate, growth_rate,
                      bn_size; drop_rate), concat_features)
              for i in 1:num_layers]
    return Chain(; layers...)
end

function dense_transition(num_input_features, num_output_features; aa_layer = false)
    pool = aa_layer ? BlurPool2d(num_output_features; stride = 2) :
           MeanPool((2, 2); stride = 2)
    return Chain(; norm = BatchNorm(num_input_features, relu),
        conv = Conv((1, 1), num_input_features => num_output_features; bias = false,
            init = Flux.kaiming_normal),
        pool)
end

@concrete struct DenseNet
    features
    global_pool
    head_drop
    classifier
end
Flux.@layer DenseNet

"""
    densenet(; growth_rate = 32, block_config = (6, 12, 24, 16), kwargs...)

Port of timm's `DenseNet` (DenseNet-BC, with the optional deep stem and blur pooling).
Keyword arguments are those of the timm class, except that `aa_layer` is `"blurpool2d"`
or `nothing`. `memory_efficient` is accepted and ignored: it only changes how gradients
are computed.
"""
function densenet(;
        growth_rate = 32, block_config = (6, 12, 24, 16), num_classes = 1000, in_chans = 3,
        global_pool = "avg", bn_size = 4, stem_type = "", act_layer = "relu",
        norm_layer = "batchnorm2d", aa_layer = nothing, drop_rate = 0.0,
        proj_drop_rate = 0.0, memory_efficient = false, aa_stem_only = true)
    check_unsupported(
        "DenseNet"; global_pool = (global_pool, "avg"), act_layer = (act_layer, "relu"),
        norm_layer = (norm_layer, "batchnorm2d"))
    aa_layer in (nothing, "blurpool2d") ||
        error("`aa_layer = $(repr(aa_layer))` is not supported by the Metalhead port of " *
              "DenseNet; use \"blurpool2d\" or nothing")
    blur = aa_layer !== nothing

    deep_stem = occursin("deep", stem_type)
    num_init_features = growth_rate * 2
    stem_pool = blur ?
                Chain(MaxPool((3, 3); stride = 1, pad = 1),
        BlurPool2d(num_init_features; stride = 2)) :
                MaxPool((3, 3); stride = 2, pad = 1)
    stem_norm(c) = BatchNorm(c, relu)
    stem_conv(args...; kwargs...) = Conv(args...; bias = false, init = Flux.kaiming_normal,
        kwargs...)
    if deep_stem
        stem_chs_1 = stem_chs_2 = growth_rate
        if occursin("tiered", stem_type)
            stem_chs_1 = 3 * (growth_rate ÷ 4)
            stem_chs_2 = occursin("narrow", stem_type) ? num_init_features :
                         6 * (growth_rate ÷ 4)
        end
        stem = [:conv0 => stem_conv((3, 3), in_chans => stem_chs_1; stride = 2, pad = 1),
            :norm0 => stem_norm(stem_chs_1),
            :conv1 => stem_conv((3, 3), stem_chs_1 => stem_chs_2; pad = 1),
            :norm1 => stem_norm(stem_chs_2),
            :conv2 => stem_conv((3, 3), stem_chs_2 => num_init_features; pad = 1),
            :norm2 => stem_norm(num_init_features),
            :pool0 => stem_pool]
    else
        stem = [:conv0 => stem_conv(
                    (7, 7), in_chans => num_init_features; stride = 2, pad = 3),
            :norm0 => stem_norm(num_init_features), :pool0 => stem_pool]
    end

    features = Pair{Symbol, Any}[stem...]
    num_features = num_init_features
    for (i, num_layers) in enumerate(block_config)
        push!(features,
            Symbol(:denseblock, i) => dense_block(
                num_layers, num_features, bn_size, growth_rate; drop_rate = proj_drop_rate))
        num_features += num_layers * growth_rate
        if i != length(block_config)
            push!(features,
                Symbol(:transition, i) => dense_transition(num_features, num_features ÷ 2;
                    aa_layer = blur && !aa_stem_only))
            num_features ÷= 2
        end
    end
    push!(features, :norm5 => stem_norm(num_features))

    pool, classifier = create_classifier(num_features, num_classes; pool_type = global_pool)
    return DenseNet(Chain(; features...), pool, dropout_or_identity(drop_rate), classifier)
end

forward_features(m::DenseNet, x) = m.features(x)
function forward_head(m::DenseNet, x; pre_logits::Bool = false)
    x = m.head_drop(m.global_pool(x))
    return pre_logits ? x : m.classifier(x)
end
(m::DenseNet)(x) = forward_head(m, forward_features(m, x))

const DENSENET_REFERENCES = Dict(
    "densenet121" =>
        ["timm/densenet121.tv_in1k" => "f0d0f2698a02cb133b09d48396db6e1e46fe9f3b"],
    "densenetblur121d" =>
        ["timm/densenetblur121d.ra_in1k" => "3ba62c856db54283ec4884b6b622d9897eddccb5"]
)

for (name, args) in [
    "densenet121" => (growth_rate = 32, block_config = (6, 12, 24, 16)),
    "densenetblur121d" => (growth_rate = 32, block_config = (6, 12, 24, 16),
        stem_type = "deep", aa_layer = "blurpool2d"),
    "densenet169" => (growth_rate = 32, block_config = (6, 12, 32, 32)),
    "densenet201" => (growth_rate = 32, block_config = (6, 12, 48, 32)),
    "densenet161" => (growth_rate = 48, block_config = (6, 12, 36, 24)),
    "densenet264d" => (growth_rate = 48, block_config = (6, 12, 64, 48), stem_type = "deep")
]
    register!(ArchSpec(TIMM, name, densenet; family = "densenet", defaults = args,
        references = get(DENSENET_REFERENCES, name, [])))
end
