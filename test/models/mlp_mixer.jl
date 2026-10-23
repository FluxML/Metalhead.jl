using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["mixer", "gmixer", "resmlp", "gmlp"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, r"(mixer|gmixer|resmlp|gmlp)_") || continue
        @test check_param_counts(arch)
    end
end

@testset "options" begin
    using Flux
    @test_throws "not supported" create_model("mixer_b16_224"; stem_norm = true)
    @test_throws "not supported" create_model("mixer_b16_224"; global_pool = "max")
    @test_throws "not supported" create_model("mixer_b16_224"; block_layer = "Block")
    @test_throws "not supported" create_model("mixer_b16_224"; mlp_layer = "GatedMlp")
    @test_throws "not supported" create_model("mixer_b16_224"; norm_layer = "batchnorm")
    @test_throws "not supported" create_model("mixer_b16_224"; act_layer = "relu")
    @test_throws "single `mlp_ratio`" create_model("resmlp_12_224"; mlp_ratio = (1, 4))

    # timm's initialisation: zero head, layer scale of the entry point, -log(n) head bias
    m = create_model("resmlp_24_224"; nlhb = true, num_classes = 10)
    @test all(iszero, m.head.weight)
    @test all(≈(-log(10f0)), m.head.bias)
    @test all(≈(1.0f-5), m.blocks.layers[1].ls1)
    @test all(≈(1.0f-6), create_model("resmlp_36_224").blocks.layers[1].ls2)
    m = create_model("mixer_s16_224"; img_size = (32, 64), num_blocks = 1)
    @test size(m.blocks.layers[1].mlp_tokens.fc1.weight, 2) == (32 ÷ 16) * (64 ÷ 16)

    m = create_model("mixer_s16_224"; img_size = 32, num_blocks = 1, num_classes = 0)
    @test size(m(rand(Float32, 32, 32, 3, 2))) == (512, 2)

    # global_pool = "" applies the head to every token
    m = create_model(fixture("gmixer"); pretrained = true)
    @test size(m(rand(Float32, 32, 16, 3, 2))) == (5, 8 * 4, 2)
end

@testset "gradient" begin
    using Flux, Functors
    for (name, channels) in ["mixer" => 2, "gmixer" => 3, "resmlp" => 3, "gmlp" => 2]
        m = Flux.trainmode!(create_model(fixture(name); pretrained = true))
        x = rand(Float32, 32, 16, channels, 2)
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        @test all(isfinite, g.stem.proj.weight)
        @test all(isfinite, g.blocks.layers[1].mlp_channels.fc1.weight)
        @test all(isfinite, g.head.weight)
        @test all(x -> !(x isa AbstractArray) || all(isfinite, x), Functors.fleaves(g))
    end
    m = Flux.trainmode!(create_model(fixture("resmlp"); pretrained = true))
    g = Flux.gradient(m -> sum(abs2, m(rand(Float32, 32, 16, 3, 2))), m)[1]
    @test all(isfinite, g.blocks.layers[1].ls1)
    @test all(isfinite, g.blocks.layers[1].norm1.alpha)
    @test all(isfinite, g.blocks.layers[1].linear_tokens.weight)
    m = Flux.trainmode!(create_model(fixture("gmlp"); pretrained = true))
    g = Flux.gradient(m -> sum(abs2, m(rand(Float32, 32, 16, 2, 2))), m)[1]
    @test all(isfinite, g.blocks.layers[1].mlp_channels.gate.proj.weight)
    @test all(isfinite, g.blocks.layers[1].mlp_channels.gate.norm.diag.scale)
end

@testset "forward_intermediates" begin
    using Flux
    m = fixture_model(fixture("resmlp"); eltype = Float32)  # patch (8, 4): a 8×2 grid
    x = rand(Float32, 32, 16, 3, 2)
    feats, nchw = forward_intermediates(m, x)
    _, nlc = forward_intermediates(m, x; output_fmt = "NLC")
    @test feats ≈ forward_features(m, x)
    @test size(nchw[1]) == (8, 2, 24, 2) && size(nlc[1]) == (24, 16, 2)
    @test nchw[2][3, 2, :, 1] == nlc[2][:, 3 + 8, 1]
    @test feature_info(m) == [(; num_chs = 24, reduction = 8, module_name = "blocks.$i")
                              for i in 0:2]
    normed = forward_intermediates(m, x; norm = true, output_fmt = "NLC",
        intermediates_only = true)
    @test normed ≈ m.norm.(nlc) && normed[3] ≈ feats
    @test forward_intermediates(m, x; indices = 1, output_fmt = "NLC")[2] ≈ nlc[3:3]
    @test forward_intermediates(m, x; indices = [3, 1], intermediates_only = true) ≈
          nchw[[1, 3]]
    early, sel = forward_intermediates(m, x; indices = [2], stop_early = true,
        output_fmt = "NLC")
    @test early ≈ m.norm(sel[1]) && sel[1] ≈ nlc[2]
    @test_throws "indices" forward_intermediates(m, x; indices = [4])
    @test_throws "output_fmt" forward_intermediates(m, x; output_fmt = "NHWC")
    loss(m) = sum(forward_intermediates(m, x; indices = [2], intermediates_only = true)[1])
    g = Flux.gradient(loss, m)[1]
    @test all(isfinite, g.stem.proj.weight)
    @test all(isfinite, g.blocks.layers[2].linear_tokens.weight)
    @test g.blocks.layers[3] === nothing
end

TEST_NETWORK && @testset "references" begin
    for prefix in ["mixer_", "resmlp_", "gmlp_"], file in reference_files(prefix)
        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x = reference_input(Tuple(JSON.parse(meta["input_shape"])))
        pre_logits = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(pre_logits, ref["pre_logits"]) <= 1e-4
    end
end
