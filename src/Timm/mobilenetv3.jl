# Ported from timm.models.mobilenetv3 (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/mobilenetv3_*, test/fixtures/lcnet_* and
# test/references/timm--*mobilenetv3*, test/references/timm--lcnet*.
#
# Blocks and builder are those of the EfficientNet family (efficientnet_blocks.jl,
# efficientnet_builder.jl). Not ported: MobileNetV4 (UIR and attention blocks, anti-
# aliasing), FBNet-V3 (`se_from_exp = false` and `rd_round_fn` work, but no fixture
# covers its tables) and feature extraction.

@concrete struct MobileNetV3
    conv_stem
    bn1
    blocks
    global_pool
    conv_head
    norm_head
    act2
    flatten
    drop
    classifier
end
Flux.@layer MobileNetV3

"""
    mobilenetv3(; block_args, num_classes = 1000, num_features = 1280, kwargs...)

Port of timm's `MobileNetV3` class (MobileNet-V3, LCNet): the `EfficientNet` stem and blocks
with the "efficient head", where global pooling comes before the head convolution and there
is no norm layer before the classifier. `num_features` is the width of the head convolution.
`block_args` comes from [`decode_arch_def`](@ref), `norm_layer` is `nothing` or the
arguments of BatchNorm2d, and `se_layer(channels; act_layer, rd_ratio)` creates the
squeeze-and-excitation modules. Other keyword arguments are those of the timm class.
"""
function mobilenetv3(; block_args, num_classes = 1000, in_chans = 3, stem_size = 16,
        fix_stem = false, num_features = 1280, head_bias = true, head_norm = false,
        pad_type = "", act_layer = nothing, norm_layer = nothing, aa_layer = nothing,
        se_layer = nothing, se_from_exp = true, round_chs_fn = round_channels,
        drop_rate = 0.0, drop_path_rate = 0.0, layer_scale_init_value = nothing,
        global_pool = "avg")
    check_unsupported("MobileNetV3"; global_pool = (global_pool, "avg"),
        aa_layer = (aa_layer, nothing),
        layer_scale_init_value = (layer_scale_init_value, nothing))
    act_layer = something(get_act_layer(act_layer), relu)
    norm_layer = resolve_norm_layer(norm_layer)
    se_layer = something(se_layer, SqueezeExcite)

    fix_stem || (stem_size = round_chs_fn(stem_size))
    conv_stem = create_conv2d(in_chans, stem_size, 3; stride = 2, padding = pad_type)
    bn1 = norm_act(norm_layer, stem_size, act_layer)

    builder = EfficientNetBuilder(; output_stride = 32, pad_type, round_chs_fn,
        se_from_exp, act_layer, norm_layer, se_layer, drop_path_rate)
    blocks = Chain(builder(stem_size, block_args))
    num_pooled_chs = builder.in_chs

    if head_norm
        # MobileNetV4: the head conv is followed by a norm+act layer
        conv_head = create_conv2d(
            num_pooled_chs, num_features, 1; padding = pad_type, bias = false)
        norm_head = norm_act(norm_layer, num_features, act_layer)
        act2 = identity
    else
        conv_head = create_conv2d(
            num_pooled_chs, num_features, 1; padding = pad_type, bias = head_bias)
        norm_head = identity
        act2 = act_layer
    end
    classifier = num_classes > 0 ? Dense(num_features => num_classes) : identity
    model = MobileNetV3(conv_stem, bn1, blocks, select_adaptive_pool2d(; flatten = false),
        conv_head, norm_head, act2, Flux.flatten, dropout_or_identity(drop_rate),
        classifier)
    return efficientnet_init_weights!(model)
end

forward_features(m::MobileNetV3, x) = m.blocks(m.bn1(m.conv_stem(x)))

function forward_head(m::MobileNetV3, x; pre_logits::Bool = false)
    x = m.act2(m.norm_head(m.conv_head(m.global_pool(x))))
    x = m.drop(m.flatten(x))
    return pre_logits ? x : m.classifier(x)
end

(m::MobileNetV3)(x) = forward_head(m, forward_features(m, x))

# timm's _create_mnv3; `bn_momentum` and `bn_eps` are resolved here, not in each _gen_*.
function create_mnv3(; features_only = false, norm_layer = nothing, bn_momentum = nothing,
        bn_eps = nothing, kwargs...)
    check_unsupported("MobileNetV3"; features_only = (features_only, false))
    norm_layer = norm_layer === nothing ? resolve_bn_args(bn_momentum, bn_eps) : norm_layer
    return mobilenetv3(; norm_layer, kwargs...)
end

# The _gen_* functions of timm, with the same keyword arguments. `variant` only selects
# timm's pretrained configuration, except in gen_mobilenet_v3, where it selects the table.

function gen_mobilenet_v3_rw(; variant = nothing, channel_multiplier = 1.0, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_nre_noskip"],
        ["ir_r1_k3_s2_e4_c24_nre", "ir_r1_k3_s1_e3_c24_nre"],
        ["ir_r3_k5_s2_e3_c40_se0.25_nre"],
        ["ir_r1_k3_s2_e6_c80", "ir_r1_k3_s1_e2.5_c80", "ir_r2_k3_s1_e2.3_c80"],
        ["ir_r2_k3_s1_e6_c112_se0.25"],
        ["ir_r3_k5_s2_e6_c160_se0.25"],
        ["cn_r1_k1_s1_c960"]
    ]
    se_layer = (channels; kw...) -> SqueezeExcite(channels; gate_layer = hardsigmoid, kw...)
    return create_mnv3(; block_args = decode_arch_def(arch_def), head_bias = false,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer = "hard_swish",
        se_layer, kwargs...)
end

function gen_mobilenet_v3(; variant, channel_multiplier = 1.0, depth_multiplier = 1.0,
        group_size = nothing, act_layer = nothing, kwargs...)
    minimal = occursin("minimal", variant)
    if occursin("small", variant)
        num_features = 1024
        if minimal
            act_layer = something(act_layer, "relu")
            arch_def = [
                ["ds_r1_k3_s2_e1_c16"],
                ["ir_r1_k3_s2_e4.5_c24", "ir_r1_k3_s1_e3.67_c24"],
                ["ir_r1_k3_s2_e4_c40", "ir_r2_k3_s1_e6_c40"],
                ["ir_r2_k3_s1_e3_c48"],
                ["ir_r3_k3_s2_e6_c96"],
                ["cn_r1_k1_s1_c576"]
            ]
        else
            act_layer = something(act_layer, "hard_swish")
            arch_def = [
                ["ds_r1_k3_s2_e1_c16_se0.25_nre"],
                ["ir_r1_k3_s2_e4.5_c24_nre", "ir_r1_k3_s1_e3.67_c24_nre"],
                ["ir_r1_k5_s2_e4_c40_se0.25", "ir_r2_k5_s1_e6_c40_se0.25"],
                ["ir_r2_k5_s1_e3_c48_se0.25"],
                ["ir_r3_k5_s2_e6_c96_se0.25"],
                ["cn_r1_k1_s1_c576"]
            ]
        end
    else
        num_features = 1280
        if minimal
            act_layer = something(act_layer, "relu")
            arch_def = [
                ["ds_r1_k3_s1_e1_c16"],
                ["ir_r1_k3_s2_e4_c24", "ir_r1_k3_s1_e3_c24"],
                ["ir_r3_k3_s2_e3_c40"],
                ["ir_r1_k3_s2_e6_c80", "ir_r1_k3_s1_e2.5_c80", "ir_r2_k3_s1_e2.3_c80"],
                ["ir_r2_k3_s1_e6_c112"],
                ["ir_r3_k3_s2_e6_c160"],
                ["cn_r1_k1_s1_c960"]
            ]
        else
            act_layer = something(act_layer, "hard_swish")
            arch_def = [
                ["ds_r1_k3_s1_e1_c16_nre"],
                ["ir_r1_k3_s2_e4_c24_nre", "ir_r1_k3_s1_e3_c24_nre"],
                ["ir_r3_k5_s2_e3_c40_se0.25_nre"],
                ["ir_r1_k3_s2_e6_c80", "ir_r1_k3_s1_e2.5_c80", "ir_r2_k3_s1_e2.3_c80"],
                ["ir_r2_k3_s1_e6_c112_se0.25"],
                ["ir_r3_k5_s2_e6_c160_se0.25"],
                ["cn_r1_k1_s1_c960"]
            ]
        end
    end
    se_layer = (channels; kw...) -> SqueezeExcite(channels; gate_layer = hardsigmoid,
        force_act_layer = relu, rd_round_fn = round_channels, kw...)
    return create_mnv3(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features, stem_size = 16, fix_stem = channel_multiplier < 0.75,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer, se_layer,
        kwargs...)
end

function gen_lcnet(; variant = nothing, channel_multiplier = 1.0, kwargs...)
    arch_def = [
        ["dsa_r1_k3_s1_c32"],
        ["dsa_r2_k3_s2_c64"],
        ["dsa_r2_k3_s2_c128"],
        ["dsa_r1_k3_s2_c256", "dsa_r1_k5_s1_c256"],
        ["dsa_r4_k5_s1_c256"],
        ["dsa_r2_k5_s2_c512_se0.25"]
    ]
    se_layer = (channels; kw...) -> SqueezeExcite(
        channels; gate_layer = hardsigmoid, force_act_layer = relu, kw...)
    return create_mnv3(; block_args = decode_arch_def(arch_def), stem_size = 16,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer = "hard_swish",
        se_layer, num_features = 1280, kwargs...)
end

const MOBILENETV3_REFERENCES = Dict(
    "mobilenetv3_large_100" => [
        "timm/mobilenetv3_large_100.ra_in1k" => "96f46a1c52932f27492dff66c72378eb99b443a7"
    ],
    "mobilenetv3_small_100" => [
        "timm/mobilenetv3_small_100.lamb_in1k" => "1824797e7887cbec1990e4adbd6675960a36c589"
    ],
    "tf_mobilenetv3_large_100" => [
        "timm/tf_mobilenetv3_large_100.in1k" => "898233b536b631ff81a48f1cb8a42d60568cb6b3"
    ],
    "tf_mobilenetv3_small_minimal_100" => [
        "timm/tf_mobilenetv3_small_minimal_100.in1k" => "0b16b3a73403a011e0961d90c072d9b30bf95a80"
    ],
    "mobilenetv3_rw" =>
        ["timm/mobilenetv3_rw.rmsp_in1k" => "d29d2e0bdcc626b1bc989aba56d45bab17eb2275"],
    "lcnet_050" => ["timm/lcnet_050.ra2_in1k" => "7d724d70bd001f453d812e733aa9b1931a69f6c2"]
)

const MOBILENETV3_ARCHS = String[]

function register_mnv3!(name, build; args...)
    push!(MOBILENETV3_ARCHS, name)
    return register!(ArchSpec(:timm, name, build; defaults = (; args...),
        references = get(MOBILENETV3_REFERENCES, name, [])))
end

let tf = (bn_eps = BN_EPS_TF_DEFAULT, pad_type = "same")
    for (name, cm, dm) in [("large_075", 0.75, 1.0), ("large_100", 1.0, 1.0),
        ("large_150d", 1.5, 1.2), ("small_050", 0.5, 1.0), ("small_075", 0.75, 1.0),
        ("small_100", 1.0, 1.0)]
        register_mnv3!("mobilenetv3_$name", gen_mobilenet_v3; variant = "mobilenetv3_$name",
            channel_multiplier = cm, depth_multiplier = dm)
    end
    for (name, cm) in [("large_075", 0.75), ("large_100", 1.0), ("large_minimal_100", 1.0),
        ("small_075", 0.75), ("small_100", 1.0), ("small_minimal_100", 1.0)]
        register_mnv3!("tf_mobilenetv3_$name", gen_mobilenet_v3;
            variant = "tf_mobilenetv3_$name", channel_multiplier = cm, tf...)
    end
    register_mnv3!("mobilenetv3_rw", gen_mobilenet_v3_rw; channel_multiplier = 1.0,
        bn_eps = BN_EPS_TF_DEFAULT)
    for (name, cm) in
        [("035", 0.35), ("050", 0.5), ("075", 0.75), ("100", 1.0), ("150", 1.5)]
        register_mnv3!("lcnet_$name", gen_lcnet; channel_multiplier = cm)
    end
end
