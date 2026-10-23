using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["efficientnet_grouped_os16", "efficientnet_tf_same", "efficientnet_edge",
        "efficientnetv2_base_tf", "mobilenetv1_head", "mobilenetv1_nohead", "mnasnet_a1",
        "mixnet_s", "mixnet_tf_same", "efficientnet_cc_tf_same", "efficientnet_cc_os16",
        "mobilenet_edgetpu_v2"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "feature extraction" begin
    for name in ["efficientnet_tf_same", "efficientnet_grouped_os16",
        "efficientnetv2_base_tf", "efficientnet_cc_os16", "mixnet_s", "mobilenetv1_nohead"]
        check_intermediate_api(name)
    end
end

@testset "architectures" begin
    # the L2 models have 480M parameters (2 GB each): their counts were checked once
    for arch in setdiff(Metalhead.Timm.EFFICIENTNET_ARCHS,
        ["efficientnet_l2", "tf_efficientnet_l2"])
        @test check_param_counts(arch)
    end
end

@testset "gradient" begin
    using Flux
    function grad(name, insize)
        m = Flux.trainmode!(create_model(fixture(name); pretrained = true))
        return Flux.gradient(m -> sum(abs2, m(rand(Float32, insize))), m)[1]
    end

    # Conv2dSame stem, depthwise-separable and inverted residual blocks with SE
    g = grad("efficientnet_tf_same", (32, 24, 3, 2))
    @test all(isfinite, g.conv_stem.conv.weight)
    @test all(isfinite, g.blocks.layers[1].layers[1].conv_dw.weight)
    @test all(isfinite, g.blocks.layers[2].layers[1].se.conv_reduce.weight)
    @test any(!iszero, g.blocks.layers[2].layers[1].se.conv_expand.weight)
    @test all(isfinite, g.classifier.weight)

    # grouped and dilated convolutions, drop path
    g = grad("efficientnet_grouped_os16", (32, 32, 1, 2))
    @test all(isfinite, g.conv_stem.weight)
    @test all(isfinite, g.blocks.layers[end].layers[1].conv_dw.weight)
    @test all(isfinite, g.conv_head.weight)

    # ConvBnAct and EdgeResidual
    g = grad("efficientnetv2_base_tf", (32, 32, 3, 2))
    @test all(isfinite, g.blocks.layers[1].layers[1].conv.weight)
    @test all(isfinite, g.blocks.layers[2].layers[1].conv_exp.conv.weight)
    @test all(isfinite, g.bn1.γ)

    # pointwise activation and head conv
    g = grad("mobilenetv1_head", (32, 32, 3, 2))
    @test all(isfinite, g.blocks.layers[1].layers[1].conv_pw.weight)
    @test all(isfinite, g.bn2.γ)

    # MixedConv2d with TF padding, CondConv2d with dynamic padding, grouped EdgeResidual
    for (name, insize) in [("mixnet_tf_same", (44, 36, 3, 2)),
        ("efficientnet_cc_tf_same", (44, 36, 3, 2)),
        ("mobilenet_edgetpu_v2", (32, 32, 3, 2))]
        m = fixture_model(fixture(name); eltype = Float32)
        @test check_gradient(m, rand(Float32, insize))
    end
end

@testset "builder" begin
    using Metalhead.Timm: decode_arch_def, decode_block_str, round_channels, get_act_layer,
                          SqueezeExcite
    using Flux

    arch_def = [["ds_r1_k3_s1_e1_c16_se0.25"], ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"], ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"], ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"]]
    depths(dm; kw...) = length.(decode_arch_def(arch_def, dm; kw...))
    # values from timm 1.0.30
    @test depths(1.0) == [1, 2, 2, 3, 3, 4, 1]
    @test depths(1.1) == [2, 3, 3, 4, 4, 5, 2]
    @test depths(1.4) == [2, 3, 3, 5, 5, 6, 2]
    @test depths(2.2) == [3, 5, 5, 7, 7, 9, 3]
    @test depths(3.6) == [4, 8, 8, 11, 11, 15, 4]
    @test depths(0.3) == [1, 1, 1, 1, 1, 2, 1]
    @test depths(1.2; depth_trunc = "round") == [1, 2, 2, 4, 4, 5, 1]
    @test depths(0.695; depth_trunc = "round") == [1, 1, 1, 2, 2, 3, 1]
    @test depths(2.2; fix_first_last = true) == [1, 5, 5, 7, 7, 9, 1]
    @test depths((1.2, 1.2, 1.2, 1.2, 1.6, 1.6, 2.0)) == [2, 3, 3, 4, 5, 7, 2]
    @test_throws "one value per stage" depths((1.2, 1.6))
    cc = decode_arch_def([["ir_r1_k3_s1_e6_c24_cc4", "ir_r1_k3_s1_e6_c24"]];
        experts_multiplier = 2)
    @test cc[1][1][:num_experts] == 8 && !haskey(cc[1][2], :num_experts)

    multi = [["ir_r1_k3_s2_e6_c24", "ir_r2_k3_s1_e3_c24", "ir_r1_k5_s1_e6_c24"],
        ["dsa_r3_k3_s1_c8", "ds_r2_k3_s1_c8"]]
    @test length.(decode_arch_def(multi, 0.5)) == [3, 3]
    @test length.(decode_arch_def(multi, 1.3)) == [6, 7]
    @test length.(decode_arch_def(multi, 1.3; depth_trunc = "round")) == [5, 6]
    @test length.(decode_arch_def(multi, 2.0)) == [8, 10]
    ba = decode_arch_def(multi, 1.0; group_size = 4)
    @test all(b -> b[:group_size] == 4, Iterators.flatten(ba))
    @test ba[2][1][:pw_act] && ba[2][1][:noskip] && !ba[2][end][:pw_act]

    args, repeats = decode_block_str("er_r2_k3_s2_e8_c32_fc24_se0.25_nre_noskip_gs8")
    @test repeats == 2
    @test args[:block_type] == "er" && args[:exp_ratio] == 8.0 && args[:stride] == 2
    @test args[:force_in_chs] == 24 && args[:se_ratio] == 0.25 && args[:noskip]
    @test args[:act_layer] === relu && args[:group_size] == 8
    args, _ = decode_block_str("ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c40_se0.5_nsw")
    @test args[:dw_kernel_size] == [3, 5, 7] && args[:exp_kernel_size] == [1, 1]
    @test args[:pw_kernel_size] == [1, 1] && args[:act_layer] === swish

    # MixedConv2d: channel splits as timm's _split_channels, depthwise when groups == in_chs
    using Metalhead.Timm: create_conv2d, MixedConv2d, CondConv2d, split_channels
    @test split_channels(48, 5) == (12, 9, 9, 9, 9) && split_channels(16, 2) == (8, 8)
    mc = create_conv2d(48, 48, [3, 5, 7, 9, 11]; stride = 2, groups = 48, padding = "same")
    @test mc isa MixedConv2d && mc.splits == (12, 9, 9, 9, 9)
    @test all(mc.convs) do c
        c isa Metalhead.Timm.Conv2dSame && c.conv.groups == size(c.conv.weight, 4)
    end
    @test first.(Metalhead.Timm.layer_params(mc))[1:2] == ["0.weight", "1.weight"]
    pw = create_conv2d(8, 24, [1, 1])
    @test size.(getfield.(pw.convs, :weight)) == ((1, 1, 4, 12), (1, 1, 4, 12))
    x = randn(Float32, 6, 6, 8, 2)
    @test pw(x) ≈ cat(pw.convs[1](x[:, :, 1:4, :]), pw.convs[2](x[:, :, 5:8, :]); dims = 3)
    @test_throws "CondConv" create_conv2d(8, 8, [3, 5]; num_experts = 4)
    @test_throws "depthwise" create_conv2d(8, 8, [3, 5]; groups = 2)

    # CondConv2d mixes the expert kernels of each sample by its routing weights
    cc = create_conv2d(4, 6, 3; num_experts = 3, stride = 2)
    @test cc isa CondConv2d && size(cc.weight) == (3 * 3 * 4 * 6, 3)
    x, r = randn(Float32, 7, 7, 4, 2), rand(Float32, 3, 2)
    y = cc(x, r)
    for b in 1:2
        conv = Conv(reshape(cc.weight * r[:, b], 3, 3, 4, 6); stride = 2, pad = 1)
        conv.weight .= reverse(conv.weight; dims = (1, 2))  # torch cross-correlates
        @test y[:, :, :, b] ≈ conv(x[:, :, :, b:b])[:, :, :, 1]
    end

    @test [round_channels(args...)
           for args in [(16, 0.25), (24, 1.1), (40, 1.2),
        (1280, 1.4), (320, 0.35), (1280, 0.35), (10, 1.0), (1792, 1.0), (272, 1.2)]] ==
          [8, 24, 48, 1792, 112, 448, 16, 1792, 328]
    @test round_channels(112, 1.8, 16) == 208
    @test round_channels(24, 0.8, 8, nothing; round_limit = 0.0) == 16
    @test round_channels(1280, 0.8, 8, nothing; round_limit = 0.0) == 1024
    @test round_channels(6, 0.75, 4, 12) == 12
    @test round_channels(50, nothing) == 50

    # activations against torch (float64) on xs
    xs = [-8.0, -3.5, -3.0, -1.0, -0.25, 0.0, 0.5, 2.9, 3.0, 5.0, 6.0, 9.0]
    expected = Dict(
        "relu" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.9, 3.0, 5.0, 6.0, 9.0],
        "relu6" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 2.9, 3.0, 5.0, 6.0, 6.0],
        "silu" => [-0.002682801043731825, -0.10259280762974712, -0.14227761953270035,
            -0.2689414213699951, -0.10945587477855048, 0.0, 0.3112296656009273,
            2.7487546670725886, 2.8577223804672998, 4.966535745378576, 5.985164261060192,
            8.998889448816124],
        "mish" => [-0.0026832508825465375, -0.10409575448637122, -0.1456474612756246,
            -0.30340146137410895, -0.12992696367914103, 0.0, 0.37524521130489513,
            2.8842668281130717, 2.9865350049679575, 4.999552077529406, 5.999926634065252,
            8.999999725928019],
        "hard_sigmoid" => [0.0, 0.0, 0.0, 0.3333333333333333, 0.4583333333333333, 0.5,
            0.5833333333333334, 0.9833333333333334, 1.0, 1.0, 1.0, 1.0],
        "hard_swish" => [0.0, 0.0, 0.0, -0.3333333333333333, -0.11458333333333333, 0.0,
            0.2916666666666667, 2.8516666666666666, 3.0, 5.0, 6.0, 9.0],
        "sigmoid" => [0.0003353501304664781, 0.02931223075135632, 0.04742587317756678,
            0.2689414213699951, 0.43782349911420193, 0.5, 0.6224593312018546,
            0.9478464369215823, 0.9525741268224334, 0.9933071490757153, 0.9975273768433653,
            0.9998766054240137],
        "gelu" => [-4.884981308350689e-15, -0.0008142017766243903, -0.00404969409489031,
            -0.15865525393145702, -0.10032341857926907, 0.0, 0.34573123063700656,
            2.8945891414288862, 2.99595030590511, 4.999998566742141, 5.999999994080474,
            9.0])
    for (name, ys) in expected
        @test get_act_layer(name).(xs) ≈ ys atol=1e-12
    end
    @test get_act_layer("swish") === get_act_layer("silu")
    @test_throws "not supported" get_act_layer("evo")

    # SE options of MobileNetV3: gate, forced activation, rounding of the reduced channels
    se = SqueezeExcite(40; rd_ratio = 0.25, act_layer = swish, gate_layer = hardsigmoid,
        force_act_layer = relu, rd_round_fn = c -> round_channels(c, 1.0))
    @test size(se.conv_reduce.weight, 4) == 16
    @test se.act1 === relu && se.gate === hardsigmoid
    @test size(SqueezeExcite(40).conv_reduce.weight, 4) == 10
    @test size(SqueezeExcite(40; rd_channels = 3).conv_reduce.weight, 4) == 3
    x = randn(Float32, 5, 5, 40, 2)
    s = se.conv_expand(relu.(se.conv_reduce(sum(x; dims = (1, 2)) ./ 25)))
    @test se(x) ≈ x .* hardsigmoid.(s)
end

@testset "initialisation" begin
    m = create_model("test_efficientnet")
    # normal(0, sqrt(2 / fan_out)) convs, uniform linear layer, zero biases
    w = m.conv_stem.weight
    @test sqrt(sum(abs2, w) / length(w))≈sqrt(2 / (3 * 3 * 24)) rtol=0.1
    @test maximum(abs, m.classifier.weight) <= 1 / sqrt(1000)
    se = m.blocks.layers[end].layers[1].se
    @test all(iszero, m.classifier.bias) && all(iszero, se.conv_reduce.bias)
    @test all(==(1), m.bn1.γ) && all(iszero, m.bn1.β)
    @test m.bn1.ϵ == 1.0f-5 && create_model("tf_efficientnet_b0").bn1.ϵ == 1.0f-3
end

@testset "drop path" begin
    # stochastic depth rate of block i (0-based) is rate * i / number of blocks
    m = create_model("efficientnet_b0"; drop_path_rate = 0.2)
    blocks = [b for stage in m.blocks.layers for b in stage.layers]
    @test length(blocks) == 16
    @test blocks[1].drop_path === identity
    @test [b.drop_path.p for b in blocks[2:end]] ≈ 0.2 .* (1:15) ./ 16
    m = create_model("efficientnet_b0")
    @test all(b -> b.drop_path === identity, [b for s in m.blocks.layers for b in s.layers])
end

@testset "unsupported options" begin
    using Metalhead.Timm: decode_block_str, EfficientNetBuilder, decode_arch_def
    using Flux: relu

    @test_throws "mha" decode_block_str("mha_r1_k3_h4_s1_d64_c160")
    @test_throws "unknown block type" decode_block_str("xx_r1_k3_s1_e6_c24")
    builder = EfficientNetBuilder(; act_layer = relu)
    @test_throws "space-to-depth" builder(
        8, decode_arch_def([["ds_r1_k3_s1_e1_c16_se0.25_d1"]]))
    @test_throws "aa_layer" EfficientNetBuilder(; act_layer = relu, aa_layer = "blurpc")(
        8, decode_arch_def([["ds_r1_k3_s1_e1_c16"]]))

    @test_throws "global_pool" create_model("efficientnet_b0"; global_pool = "max")
    @test_throws "aa_layer" create_model("efficientnet_b0"; aa_layer = "blurpc")
    @test_throws "features_only" create_model("efficientnet_b0"; features_only = true)
    @test_throws "norm_layer" create_model("efficientnet_b0"; norm_layer = "groupnorm")
    @test_throws "not supported" create_model("efficientnet_b0"; act_layer = "evo")
    @test_throws "neither a ported architecture" create_model("efficientnet_x_b5")
    @test_throws "neither a ported architecture" create_model("efficientnet_b1_pruned")
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("efficientnet") ∪ reference_files("tf_efficientnet") ∪
                reference_files("mobilenetv1") ∪ reference_files("mobilenetv2") ∪
                reference_files("mnasnet") ∪ reference_files("test_efficientnet") ∪
                reference_files("mixnet") ∪ reference_files("tf_mixnet") ∪
                reference_files("fbnetc") ∪ reference_files("mobilenet_edgetpu")

        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        pre_logits = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(pre_logits, ref["pre_logits"]) <= 1e-4
    end
end
