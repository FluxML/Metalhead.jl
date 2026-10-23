@testitem "Models: timm convmixer fixtures" setup=[ParitySetup] begin
    for name in ["convmixer", "convmixer_even_kernel"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testitem "Models: timm convmixer architectures" setup=[ParitySetup] begin
    for arch in list_models()
        startswith(arch, "convmixer_") || continue
        @test check_param_counts(arch)
    end
end

@testitem "Models: timm convmixer options" setup=[ParitySetup] begin
    @test_throws "not supported" create_model("convmixer_768_32"; act_layer = "silu")
    @test_throws "not supported" create_model("convmixer_768_32"; global_pool = "")
    @test_throws "not supported" create_model("convmixer_768_32"; global_pool = "max")

    m = create_model("convmixer_768_32"; dim = 8, depth = 1, num_classes = 0)
    @test size(m(rand(Float32, 16, 16, 3, 2))) == (8, 2)
end

@testitem "Models: timm convmixer gradient" setup=[ParitySetup] begin
    using Flux, Functors
    # the even kernel is a Conv2dSame, which wraps its Conv
    inputs = ["convmixer" => (32, 16, 2, 2), "convmixer_even_kernel" => (16, 16, 3, 2)]
    for (name, input) in inputs
        m = Flux.trainmode!(create_model(fixture(name); pretrained = true))
        x = rand(Float32, input...)
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        depthwise = g.blocks.layers[1].layers[1].fn.layers[1]
        @test all(isfinite, name == "convmixer" ? depthwise.weight : depthwise.conv.weight)
        @test all(isfinite, g.stem.layers[1].weight)
        @test all(isfinite, g.blocks.layers[2].layers[2].weight)
        @test all(isfinite, g.head.weight)
        @test all(x -> !(x isa AbstractArray) || all(isfinite, x), Functors.fleaves(g))
    end
end

@testitem "Models: timm convmixer references" tags=[:network] setup=[ParitySetup] begin
    for file in reference_files("convmixer_")
        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x = reference_input(Tuple(JSON.parse(meta["input_shape"])))
        pre_logits = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(pre_logits, ref["pre_logits"]) <= 1e-4
    end
end
