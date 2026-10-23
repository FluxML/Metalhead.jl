@testitem "Models: timm convnext fixtures" setup=[ParitySetup] begin
    for name in ["convnext_conv_mlp", "convnext_linear_mlp"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testitem "Models: timm convnext architectures" setup=[ParitySetup] begin
    for arch in list_models()
        startswith(arch, "convnext") || continue
        @test check_param_counts(arch)
    end
end

@testitem "Models: timm convnext gradient" setup=[ParitySetup] begin
    using Flux
    x = rand(Float32, 64, 64, 3, 2)
    for name in ["convnext_conv_mlp", "convnext_linear_mlp"]
        m = Flux.trainmode!(create_model(fixture(name); pretrained = true))
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        @test all(isfinite, g.stem.layers[1].weight)
        @test all(isfinite, g.stages.layers[2].blocks.layers[1].gamma)
        @test all(isfinite, g.stages.layers[4].blocks.layers[1].conv_dw.weight)
        @test all(isfinite, g.head.norm.weight)
        @test all(isfinite, g.head.fc.weight)
    end
end

@testitem "Models: timm convnext references" tags=[:network] setup=[ParitySetup] begin
    for file in reference_files("convnext")
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true), ref["pre_logits"]) <=
              1e-4
    end
end
