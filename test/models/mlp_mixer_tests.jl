@testitem "Models: timm mlp_mixer fixtures" setup=[ParitySetup] begin
    for name in ["mixer", "gmixer", "resmlp", "gmlp"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testitem "Models: timm mlp_mixer architectures" setup=[ParitySetup] begin
    for arch in list_models()
        startswith(arch, r"(mixer|gmixer|resmlp|gmlp)_") || continue
        @test check_param_counts(arch)
    end
end

@testitem "Models: timm mlp_mixer options" setup=[ParitySetup] begin
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

@testitem "Models: timm mlp_mixer gradient" setup=[ParitySetup] begin
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

@testitem "Models: timm mlp_mixer references" tags=[:network] setup=[ParitySetup] begin
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
