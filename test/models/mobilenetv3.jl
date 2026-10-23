using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in [
        "mobilenetv3_large", "mobilenetv3_tf_large", "mobilenetv3_small_head_norm",
        "mobilenetv3_tf_small_minimal", "mobilenetv3_rw", "lcnet", "fbnetv3"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "feature extraction" begin
    for name in [
        "mobilenetv3_large", "mobilenetv3_small_head_norm", "mobilenetv4_conv_small",
        "mobilenetv4_hybrid_medium"]
        check_intermediate_api(name)
    end
end

@testset "architectures" begin
    for arch in Metalhead.Timm.MOBILENETV3_ARCHS
        @test check_param_counts(arch)
    end
end

@testset "gradient" begin
    for (name, insize) in [("mobilenetv3_large", (32, 32, 3, 2)),
        ("mobilenetv3_tf_large", (48, 32, 3, 2)),
        ("mobilenetv3_small_head_norm", (32, 32, 1, 2)), ("mobilenetv3_rw", (32, 32, 3, 2)),
        ("lcnet", (32, 32, 3, 2)), ("fbnetv3", (32, 32, 3, 2))]
        m = fixture_model(fixture(name); eltype = Float32)
        @test check_gradient(m, rand(Float32, insize))
    end
end

@testset "head and options" begin
    using Flux

    # the stem of the models below 0.75 width is not scaled (`fix_stem`), LCNet's is
    stem(arch) = size(create_model(arch).conv_stem.weight, 4)
    @test stem("mobilenetv3_small_050") == 16
    @test stem("lcnet_050") == 8
    @test stem("mobilenetv3_large_075") == 16

    # efficient head: pooled features go through the head conv; the bias is optional
    m = create_model("mobilenetv3_large_100")
    @test size(m.conv_head.weight) == (1, 1, 960, 1280)
    @test m.conv_head.bias isa AbstractArray
    @test m.norm_head === identity && m.act2 === hardswish
    @test create_model("mobilenetv3_rw").conv_head.bias === false
    @test create_model("lcnet_050").act2 === hardswish
    h = create_model("mobilenetv3_small_050"; head_norm = true)
    @test h.conv_head.bias === false && h.act2 === identity && h.norm_head isa BatchNorm
    features = forward_features(m, rand(Float32, 64, 64, 3, 2))
    @test size(features) == (2, 2, 960, 2)
    @test size(forward_head(m, features; pre_logits = true)) == (1280, 2)
    @test size(forward_head(m, features)) == (1000, 2)

    # SE channels come from the expanded channels (`se_from_exp`) and are rounded to 8,
    # as in timm: blocks.0.0 and blocks.2.1 of mobilenetv3_small_050
    s = create_model("mobilenetv3_small_050")
    se(m, i, j) = m.blocks.layers[i].layers[j].se
    @test size(se(s, 1, 1).conv_reduce.weight) == (1, 1, 16, 8)
    @test size(se(s, 3, 2).conv_reduce.weight) == (1, 1, 144, 40)
    @test se(s, 1, 1).gate === hardsigmoid && se(s, 1, 1).act1 === relu
    # without it, the reduction is divided by the expansion ratio
    t = create_model("mobilenetv3_small_050"; se_from_exp = false)
    @test size(se(t, 3, 2).conv_reduce.weight, 3) == 144
    @test size(se(t, 3, 2).conv_reduce.weight, 4) < 40

    # the TF variants
    tf = create_model("tf_mobilenetv3_large_100")
    @test tf.bn1.ϵ == 1.0f-3 && tf.conv_stem isa Metalhead.Timm.Conv2dSame
    @test create_model("mobilenetv3_rw").bn1.ϵ == 1.0f-3
    @test create_model("mobilenetv3_large_100").bn1.ϵ == 1.0f-5
end

@testset "MobileNetV4 fixtures" begin
    for name in ["mobilenetv4_conv_small", "mobilenetv4_conv_blur_medium",
        "mobilenetv4_conv_aa_large", "mobilenetv4_hybrid_medium",
        "mobilenetv4_hybrid_large"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "MobileNetV4 gradient" begin
    for (name, insize) in [("mobilenetv4_conv_blur_medium", (40, 48, 1, 2)),
        ("mobilenetv4_hybrid_medium", (48, 64, 1, 2))]
        m = fixture_model(fixture(name); eltype = Float32)
        @test check_gradient(m, rand(Float32, insize))
    end
end

@testset "MobileNetV4 options" begin
    using Flux
    T = Metalhead.Timm
    block(m, i, j) = m.blocks.layers[i].layers[j]

    # the head conv is followed by a norm layer, and the stem is fixed below width 1
    m = create_model("mobilenetv4_conv_small")
    @test m.conv_head.bias === false && m.norm_head isa BatchNorm && m.act2 === identity
    @test size(m.conv_stem.weight, 4) == 32
    @test size(create_model("mobilenetv4_conv_small_050").conv_stem.weight, 4) == 32
    @test size(forward_features(m, rand(Float32, 64, 64, 3, 1))) == (2, 2, 960, 1)

    # UIR variants: ExtraDW, IR (no dw_start), ConvNeXt (no dw_mid), FFN (neither)
    c = create_model("mobilenetv4_conv_medium")
    @test block(c, 2, 1) isa T.UniversalInvertedResidual
    @test block(c, 3, 7).dw_start === identity && block(c, 3, 7).dw_mid === identity
    @test block(c, 3, 6).dw_mid === identity && block(c, 3, 6).dw_start isa Chain
    @test block(c, 2, 1).layer_scale === identity
    @test create_model("mobilenetv4_hybrid_large").bn1.λ === gelu_erf

    # hybrids: multi-query attention with downsampled keys/values in stage 3, layer scale
    h = create_model("mobilenetv4_hybrid_medium")
    a = block(h, 3, 5)
    @test a isa T.MobileAttention && a.layer_scale isa T.LayerScale2d
    @test a.attn.num_heads == 4 && haskey(a.attn.key.layers, :down_conv)
    @test !haskey(block(h, 4, 9).attn.key.layers, :down_conv)
    @test block(h, 3, 4).layer_scale isa T.LayerScale2d

    # anti-aliasing replaces the stride of the strided convs
    aa = create_model("mobilenetv4_conv_aa_large")
    @test block(aa, 1, 1).aa isa MeanPool && block(aa, 1, 1).conv_exp.stride == (1, 1)
    @test block(aa, 2, 1).dw_mid.layers.aa isa MeanPool
    @test !haskey(block(aa, 2, 2).dw_mid.layers, :aa)
    b = create_model("mobilenetv4_conv_blur_medium")
    @test block(b, 2, 1).dw_mid.layers.aa isa T.BlurPool2d
    @test block(b, 2, 1).dw_mid.layers.aa.pad_mode == "constant"

    @test_throws "mha" T.decode_arch_def([["mha_r1_k3_h4_s1_d64_c160"]])
    @test_throws "mha" T.MobileAttention(16, 16)
    @test_throws "query_strides" T.MultiQueryAttention2d(16; query_strides = 2)
end

@testset "unsupported options" begin
    @test_throws "global_pool" create_model("mobilenetv3_large_100"; global_pool = "max")
    @test_throws "aa_layer" create_model("mobilenetv3_large_100"; aa_layer = "avg")
    @test_throws "features_only" create_model("mobilenetv3_large_100"; features_only = true)
    @test_throws "norm_layer" create_model(
        "mobilenetv3_large_100"; norm_layer = "groupnorm")
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("mobilenetv3") ∪ reference_files("tf_mobilenetv3") ∪
                reference_files("lcnet") ∪ reference_files("fbnetv3") ∪
                reference_files("mobilenetv4")

        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x = reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        pre_logits = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(pre_logits, ref["pre_logits"]) <= 1e-4
    end
end
