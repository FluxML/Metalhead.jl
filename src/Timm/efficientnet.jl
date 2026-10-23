# Ported from timm.models.efficientnet (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/efficientnet_*, test/fixtures/mobilenet*,
# test/fixtures/mnasnet* and test/references/timm--*.
#
# Not ported: CondConv, MixNet, EfficientNet-X/H, GroupNorm/EvoNorm variants, the blur and
# global-context variants, feature extraction, and the registered architectures that need
# them. `fbnetc_100` is missing too: timm builds it with `bn_eps = 1e-3` only when loading
# pretrained weights, and a Hub config does not say so.

@concrete struct EfficientNet
    conv_stem
    bn1
    blocks
    conv_head
    bn2
    global_pool
    drop
    classifier
end
Flux.@layer EfficientNet

"""
    efficientnet(; block_args, num_classes = 1000, num_features = 1280, kwargs...)

Port of timm's `EfficientNet` class (EfficientNet, EfficientNet-V2, MobileNet-V1/V2, MNASNet
and the others built by `timm.models.efficientnet`). `block_args` comes from
[`decode_arch_def`](@ref); `norm_layer` is `nothing` or the arguments of BatchNorm2d, and
`se_layer(channels; act_layer, rd_ratio)` creates the squeeze-and-excitation modules.
Other keyword arguments are those of the timm class.
"""
function efficientnet(; block_args, num_classes = 1000, num_features = 1280, in_chans = 3,
        stem_size = 32, stem_kernel_size = 3, fix_stem = false, output_stride = 32,
        pad_type = "", act_layer = nothing, norm_layer = nothing, aa_layer = nothing,
        se_layer = nothing, round_chs_fn = round_channels, drop_rate = 0.0,
        drop_path_rate = 0.0, global_pool = "avg")
    check_unsupported("EfficientNet"; global_pool = (global_pool, "avg"),
        aa_layer = (aa_layer, nothing))
    act_layer = something(get_act_layer(act_layer), relu)
    norm_layer = resolve_norm_layer(norm_layer)
    se_layer = something(se_layer, SqueezeExcite)

    fix_stem || (stem_size = round_chs_fn(stem_size))
    conv_stem = create_conv2d(in_chans, stem_size, stem_kernel_size; stride = 2,
        padding = pad_type)
    bn1 = norm_act(norm_layer, stem_size, act_layer)

    builder = EfficientNetBuilder(; output_stride, pad_type, round_chs_fn, act_layer,
        norm_layer, se_layer, drop_path_rate)
    blocks = Chain(builder(stem_size, block_args))
    head_chs = builder.in_chs

    if num_features > 0
        conv_head = create_conv2d(head_chs, num_features, 1; padding = pad_type)
        bn2 = norm_act(norm_layer, num_features, act_layer)
    else
        conv_head, bn2, num_features = identity, identity, head_chs
    end
    global_pool, classifier = create_classifier(num_features, num_classes)
    model = EfficientNet(conv_stem, bn1, blocks, conv_head, bn2, global_pool,
        dropout_or_identity(drop_rate), classifier)
    return efficientnet_init_weights!(model)
end

function forward_features(m::EfficientNet, x)
    return m.bn2(m.conv_head(m.blocks(m.bn1(m.conv_stem(x)))))
end

function forward_head(m::EfficientNet, x; pre_logits::Bool = false)
    x = m.drop(m.global_pool(x))
    return pre_logits ? x : m.classifier(x)
end

(m::EfficientNet)(x) = forward_head(m, forward_features(m, x))

# timm's _create_effnet; `bn_momentum` and `bn_eps` are resolved here, not in each _gen_*.
function create_effnet(; features_only = false, norm_layer = nothing, bn_momentum = nothing,
        bn_eps = nothing, kwargs...)
    check_unsupported("EfficientNet"; features_only = (features_only, false))
    norm_layer = norm_layer === nothing ? resolve_bn_args(bn_momentum, bn_eps) : norm_layer
    return efficientnet(; norm_layer, kwargs...)
end

# The _gen_* functions of timm, with the same keyword arguments. `variant` only selects
# timm's pretrained configuration and is ignored.

function gen_mnasnet_a1(; variant = nothing, channel_multiplier = 1.0, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_noskip"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r3_k5_s2_e3_c40_se0.25"],
        ["ir_r4_k3_s2_e6_c80"],
        ["ir_r2_k3_s1_e6_c112_se0.25"],
        ["ir_r3_k5_s2_e6_c160_se0.25"],
        ["ir_r1_k3_s1_e6_c320"]
    ]
    return create_effnet(; block_args = decode_arch_def(arch_def), stem_size = 32,
        round_chs_fn = c -> round_channels(c, channel_multiplier), kwargs...)
end

function gen_mnasnet_b1(; variant = nothing, channel_multiplier = 1.0, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_c16_noskip"],
        ["ir_r3_k3_s2_e3_c24"],
        ["ir_r3_k5_s2_e3_c40"],
        ["ir_r3_k5_s2_e6_c80"],
        ["ir_r2_k3_s1_e6_c96"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320_noskip"]
    ]
    return create_effnet(; block_args = decode_arch_def(arch_def), stem_size = 32,
        round_chs_fn = c -> round_channels(c, channel_multiplier), kwargs...)
end

function gen_mnasnet_small(; variant = nothing, channel_multiplier = 1.0, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_c8"],
        ["ir_r1_k3_s2_e3_c16"],
        ["ir_r2_k3_s2_e6_c16"],
        ["ir_r4_k5_s2_e6_c32_se0.25"],
        ["ir_r3_k3_s1_e6_c32_se0.25"],
        ["ir_r3_k5_s2_e6_c88_se0.25"],
        ["ir_r1_k3_s1_e6_c144"]
    ]
    return create_effnet(; block_args = decode_arch_def(arch_def), stem_size = 8,
        round_chs_fn = c -> round_channels(c, channel_multiplier), kwargs...)
end

function gen_mobilenet_v1(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, fix_stem_head = false,
        head_conv = false, kwargs...)
    arch_def = [
        ["dsa_r1_k3_s1_c64"],
        ["dsa_r2_k3_s2_c128"],
        ["dsa_r2_k3_s2_c256"],
        ["dsa_r6_k3_s2_c512"],
        ["dsa_r2_k3_s2_c1024"]
    ]
    round_chs_fn = c -> round_channels(c, channel_multiplier)
    head_features = head_conv ?
                    (fix_stem_head ? 1024 : max(1024, round_chs_fn(1024))) : 0
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier;
            fix_first_last = fix_stem_head, group_size),
        num_features = head_features, stem_size = 32, fix_stem = fix_stem_head,
        round_chs_fn, act_layer = "relu6", kwargs...)
end

function gen_mobilenet_v2(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, fix_stem_head = false, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_c16"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r3_k3_s2_e6_c32"],
        ["ir_r4_k3_s2_e6_c64"],
        ["ir_r3_k3_s1_e6_c96"],
        ["ir_r3_k3_s2_e6_c160"],
        ["ir_r1_k3_s1_e6_c320"]
    ]
    round_chs_fn = c -> round_channels(c, channel_multiplier)
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier;
            fix_first_last = fix_stem_head, group_size),
        num_features = fix_stem_head ? 1280 : max(1280, round_chs_fn(1280)),
        stem_size = 32, fix_stem = fix_stem_head, round_chs_fn, act_layer = "relu6",
        kwargs...)
end

function gen_spnasnet(; variant = nothing, channel_multiplier = 1.0, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_c16_noskip"],
        ["ir_r3_k3_s2_e3_c24"],
        ["ir_r1_k5_s2_e6_c40", "ir_r3_k3_s1_e3_c40"],
        ["ir_r1_k5_s2_e6_c80", "ir_r3_k3_s1_e3_c80"],
        ["ir_r1_k5_s1_e6_c96", "ir_r3_k5_s1_e3_c96"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320_noskip"]
    ]
    return create_effnet(; block_args = decode_arch_def(arch_def), stem_size = 32,
        round_chs_fn = c -> round_channels(c, channel_multiplier), kwargs...)
end

function gen_efficientnet(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, channel_divisor = 8, group_size = nothing, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"]
    ]
    round_chs_fn = c -> round_channels(c, channel_multiplier, channel_divisor)
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = round_chs_fn(1280), stem_size = 32, round_chs_fn,
        act_layer = "swish", kwargs...)
end

function gen_efficientnet_edge(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, kwargs...)
    arch_def = [
        ["er_r1_k3_s1_e4_c24_fc24_noskip"],
        ["er_r2_k3_s2_e8_c32"],
        ["er_r4_k3_s2_e8_c48"],
        ["ir_r5_k5_s2_e8_c96"],
        ["ir_r4_k5_s1_e8_c144"],
        ["ir_r2_k5_s2_e8_c192"]
    ]
    round_chs_fn = c -> round_channels(c, channel_multiplier)
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = round_chs_fn(1280), stem_size = 32, round_chs_fn,
        act_layer = "relu", kwargs...)
end

function gen_efficientnet_lite(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_e1_c16"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r2_k5_s2_e6_c40"],
        ["ir_r3_k3_s2_e6_c80"],
        ["ir_r3_k5_s1_e6_c112"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320"]
    ]
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; fix_first_last = true),
        num_features = 1280, stem_size = 32, fix_stem = true,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer = "relu6",
        kwargs...)
end

function gen_efficientnetv2_base(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, kwargs...)
    arch_def = [
        ["cn_r1_k3_s1_e1_c16_skip"],
        ["er_r2_k3_s2_e4_c32"],
        ["er_r2_k3_s2_e4_c48"],
        ["ir_r3_k3_s2_e4_c96_se0.25"],
        ["ir_r5_k3_s1_e6_c112_se0.25"],
        ["ir_r8_k3_s2_e6_c192_se0.25"]
    ]
    round_chs_fn = c -> round_channels(c, channel_multiplier; round_limit = 0.0)
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = round_chs_fn(1280), stem_size = 32, round_chs_fn,
        act_layer = "silu", kwargs...)
end

function gen_efficientnetv2_s(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, rw = false, kwargs...)
    arch_def = [
        ["cn_r2_k3_s1_e1_c24_skip"],
        ["er_r4_k3_s2_e4_c48"],
        ["er_r4_k3_s2_e4_c64"],
        ["ir_r6_k3_s2_e4_c128_se0.25"],
        ["ir_r9_k3_s1_e6_c160_se0.25"],
        ["ir_r15_k3_s2_e6_c256_se0.25"]
    ]
    num_features = 1280
    if rw
        arch_def[1] = ["er_r2_k3_s1_e1_c24"]
        arch_def[end] = ["ir_r15_k3_s2_e6_c272_se0.25"]
        num_features = 1792
    end
    round_chs_fn = c -> round_channels(c, channel_multiplier)
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = round_chs_fn(num_features), stem_size = 24, round_chs_fn,
        act_layer = "silu", kwargs...)
end

function gen_efficientnetv2_m(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, kwargs...)
    arch_def = [
        ["cn_r3_k3_s1_e1_c24_skip"],
        ["er_r5_k3_s2_e4_c48"],
        ["er_r5_k3_s2_e4_c80"],
        ["ir_r7_k3_s2_e4_c160_se0.25"],
        ["ir_r14_k3_s1_e6_c176_se0.25"],
        ["ir_r18_k3_s2_e6_c304_se0.25"],
        ["ir_r5_k3_s1_e6_c512_se0.25"]
    ]
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = 1280, stem_size = 24,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer = "silu",
        kwargs...)
end

function gen_efficientnetv2_l(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, kwargs...)
    arch_def = [
        ["cn_r4_k3_s1_e1_c32_skip"],
        ["er_r7_k3_s2_e4_c64"],
        ["er_r7_k3_s2_e4_c96"],
        ["ir_r10_k3_s2_e4_c192_se0.25"],
        ["ir_r19_k3_s1_e6_c224_se0.25"],
        ["ir_r25_k3_s2_e6_c384_se0.25"],
        ["ir_r7_k3_s1_e6_c640_se0.25"]
    ]
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = 1280, stem_size = 32,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer = "silu",
        kwargs...)
end

function gen_efficientnetv2_xl(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, group_size = nothing, kwargs...)
    arch_def = [
        ["cn_r4_k3_s1_e1_c32_skip"],
        ["er_r8_k3_s2_e4_c64"],
        ["er_r8_k3_s2_e4_c96"],
        ["ir_r16_k3_s2_e4_c192_se0.25"],
        ["ir_r24_k3_s1_e6_c256_se0.25"],
        ["ir_r32_k3_s2_e6_c512_se0.25"],
        ["ir_r8_k3_s1_e6_c640_se0.25"]
    ]
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; group_size),
        num_features = 1280, stem_size = 32,
        round_chs_fn = c -> round_channels(c, channel_multiplier), act_layer = "silu",
        kwargs...)
end

function gen_tinynet(; variant = nothing, model_width = 1.0, depth_multiplier = 1.0,
        kwargs...)
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"], ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"], ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"], ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"]
    ]
    return create_effnet(;
        block_args = decode_arch_def(arch_def, depth_multiplier; depth_trunc = "round"),
        num_features = max(1280, round_channels(1280, model_width, 8, nothing)),
        stem_size = 32, fix_stem = true,
        round_chs_fn = c -> round_channels(c, model_width), act_layer = "swish", kwargs...)
end

function gen_test_efficientnet(; variant = nothing, channel_multiplier = 1.0,
        depth_multiplier = 1.0, kwargs...)
    arch_def = [
        ["cn_r1_k3_s1_e1_c16_skip"],
        ["er_r1_k3_s2_e4_c24"],
        ["er_r1_k3_s2_e4_c32"],
        ["ir_r1_k3_s2_e4_c48_se0.25"],
        ["ir_r1_k3_s2_e4_c64_se0.25"]
    ]
    round_chs_fn = c -> round_channels(c, channel_multiplier; round_limit = 0.0)
    return create_effnet(; block_args = decode_arch_def(arch_def, depth_multiplier),
        num_features = round_chs_fn(256), stem_size = 24, round_chs_fn,
        act_layer = "silu", kwargs...)
end

const EFFICIENTNET_REFERENCES = Dict(
    "efficientnet_b0" =>
        ["timm/efficientnet_b0.ra_in1k" => "1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a"],
    "tf_efficientnet_b0" => [
        "timm/tf_efficientnet_b0.ns_jft_in1k" => "625afb926af7ee851c12bdfef4520d7b2ba679d5"
    ],
    "mobilenetv1_100" => [
        "timm/mobilenetv1_100.ra4_e3600_r224_in1k" => "6e88b523abded44469eafcd52f7832d5e0300b1b"
    ],
    "mobilenetv2_100" =>
        ["timm/mobilenetv2_100.ra_in1k" => "5afa12513b048c79b9147a5d210e9bdc50035481"],
    "mnasnet_100" =>
        ["timm/mnasnet_100.rmsp_in1k" => "a30af72360c7a4871b0156cc11d7b767208273d7"],
    "tf_efficientnetv2_b0" =>
        ["timm/tf_efficientnetv2_b0.in1k" => "30b1df5883752d711e047538d9b01db797782818"],
    "test_efficientnet" =>
        ["timm/test_efficientnet.r160_in1k" => "8e9bd84a0b17d27dd068b461fe255b94543d073b"]
)

const EFFICIENTNET_ARCHS = String[]

function register_effnet!(name, build; args...)
    push!(EFFICIENTNET_ARCHS, name)
    return register!(ArchSpec(:timm, name, build; defaults = (; args...),
        references = get(EFFICIENTNET_REFERENCES, name, [])))
end

let tf = (bn_eps = BN_EPS_TF_DEFAULT, pad_type = "same")
    for (name, cm) in ["mnasnet_050" => 0.5, "mnasnet_075" => 0.75, "mnasnet_100" => 1.0,
        "mnasnet_140" => 1.4]
        register_effnet!(name, gen_mnasnet_b1; channel_multiplier = cm)
    end
    for (name, cm) in ["semnasnet_050" => 0.5, "semnasnet_075" => 0.75,
        "semnasnet_100" => 1.0, "semnasnet_140" => 1.4]
        register_effnet!(name, gen_mnasnet_a1; channel_multiplier = cm)
    end
    register_effnet!("mnasnet_small", gen_mnasnet_small; channel_multiplier = 1.0)
    register_effnet!("spnasnet_100", gen_spnasnet; channel_multiplier = 1.0)

    register_effnet!("mobilenetv1_100", gen_mobilenet_v1; channel_multiplier = 1.0)
    register_effnet!("mobilenetv1_100h", gen_mobilenet_v1; channel_multiplier = 1.0,
        head_conv = true)
    register_effnet!("mobilenetv1_125", gen_mobilenet_v1; channel_multiplier = 1.25)
    for (name, cm) in ["mobilenetv2_035" => 0.35, "mobilenetv2_050" => 0.5,
        "mobilenetv2_075" => 0.75, "mobilenetv2_100" => 1.0, "mobilenetv2_140" => 1.4]
        register_effnet!(name, gen_mobilenet_v2; channel_multiplier = cm)
    end
    register_effnet!("mobilenetv2_110d", gen_mobilenet_v2; channel_multiplier = 1.1,
        depth_multiplier = 1.2, fix_stem_head = true)
    register_effnet!("mobilenetv2_120d", gen_mobilenet_v2; channel_multiplier = 1.2,
        depth_multiplier = 1.4, fix_stem_head = true)

    for (name, width, depth) in [("tinynet_a", 1.0, 1.2), ("tinynet_b", 0.75, 1.1),
        ("tinynet_c", 0.54, 0.85), ("tinynet_d", 0.54, 0.695), ("tinynet_e", 0.51, 0.6)]
        register_effnet!(name, gen_tinynet; model_width = width, depth_multiplier = depth)
    end

    for (i, cm, dm) in [(0, 1.0, 1.0), (1, 1.0, 1.1), (2, 1.1, 1.2), (3, 1.2, 1.4),
        (4, 1.4, 1.8), (5, 1.6, 2.2), (6, 1.8, 2.6), (7, 2.0, 3.1), (8, 2.2, 3.6)]
        register_effnet!("efficientnet_b$i", gen_efficientnet; channel_multiplier = cm,
            depth_multiplier = dm)
        register_effnet!("tf_efficientnet_b$i", gen_efficientnet; channel_multiplier = cm,
            depth_multiplier = dm, tf...)
    end
    for (i, cm, dm) in [(0, 1.0, 1.0), (1, 1.0, 1.1), (2, 1.1, 1.2), (3, 1.2, 1.4),
        (4, 1.4, 1.8)]
        register_effnet!("efficientnet_lite$i", gen_efficientnet_lite;
            channel_multiplier = cm, depth_multiplier = dm)
        register_effnet!("tf_efficientnet_lite$i", gen_efficientnet_lite;
            channel_multiplier = cm, depth_multiplier = dm, tf...)
    end
    for (name, cm, dm) in [("es", 1.0, 1.0), ("es_pruned", 1.0, 1.0), ("em", 1.0, 1.1),
        ("el", 1.2, 1.4), ("el_pruned", 1.2, 1.4)]
        register_effnet!("efficientnet_$name", gen_efficientnet_edge;
            channel_multiplier = cm, depth_multiplier = dm)
    end
    for (name, cm, dm) in [("es", 1.0, 1.0), ("em", 1.0, 1.1), ("el", 1.2, 1.4)]
        register_effnet!("tf_efficientnet_$name", gen_efficientnet_edge;
            channel_multiplier = cm, depth_multiplier = dm, tf...)
    end

    register_effnet!("efficientnetv2_rw_t", gen_efficientnetv2_s; channel_multiplier = 0.8,
        depth_multiplier = 0.9, rw = false)
    register_effnet!("efficientnetv2_rw_s", gen_efficientnetv2_s; rw = true)
    register_effnet!("efficientnetv2_rw_m", gen_efficientnetv2_s; channel_multiplier = 1.2,
        depth_multiplier = (1.2, 1.2, 1.2, 1.2, 1.6, 1.6), rw = true)
    for (name, gen) in [("s", gen_efficientnetv2_s), ("m", gen_efficientnetv2_m),
        ("l", gen_efficientnetv2_l), ("xl", gen_efficientnetv2_xl)]
        register_effnet!("efficientnetv2_$name", gen)
        register_effnet!("tf_efficientnetv2_$name", gen; tf...)
    end
    for (i, cm, dm) in [(0, 1.0, 1.0), (1, 1.0, 1.1), (2, 1.1, 1.2), (3, 1.2, 1.4)]
        register_effnet!("tf_efficientnetv2_b$i", gen_efficientnetv2_base;
            channel_multiplier = cm, depth_multiplier = dm, tf...)
    end
    register_effnet!("test_efficientnet", gen_test_efficientnet)
end
