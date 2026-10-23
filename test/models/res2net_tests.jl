@testitem "Models: timm res2net fixtures" setup=[ParitySetup] begin
    for name in ["res2net", "res2next_deep"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testitem "Models: timm res2net architectures" setup=[ParitySetup] begin
    for arch in list_models()
        startswith(arch, "res2ne") || continue
        @test check_param_counts(arch)
    end
end

@testitem "Models: timm res2net gradient" setup=[ParitySetup] begin
    using Flux
    m = Flux.trainmode!(create_model(fixture("res2net"); pretrained = true))
    x = rand(Float32, 32, 32, 3, 2)
    g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
    for block in (g.layer1.layers[1], g.layer1.layers[2])
        @test all(isfinite, block.conv1.weight)
        @test all(!all(iszero, c.weight) && all(isfinite, c.weight) for c in block.convs)
    end
    @test all(isfinite, g.conv1.weight)
end

@testitem "Models: timm res2net block arguments" begin
    using Metalhead: create_model

    tiny = (; layers = (1, 1, 1, 1), channels = (8, 16, 32, 64), num_classes = 10)
    block(; kw...) = create_model("res2net50_26w_4s"; tiny..., kw...).layer1.layers[1]
    @test length(block().convs) == 3
    @test length(block(block_args = (scale = 6,)).convs) == 5
    @test length(block(block_args = Dict("scale" => 2)).convs) == 1
    @test_throws "block_args" block(block_args = (attn_layer = "se",))
    @test_throws "block_args" block(block_args = (width = 2,))
end

@testitem "Models: timm res2net references" tags=[:network] setup=[ParitySetup] begin
    for file in reference_files("res2ne")
        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x = reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        features = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(features, ref["pre_logits"]) <= 1e-4
    end
end
