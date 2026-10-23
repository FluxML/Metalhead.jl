@testitem "Models: timm resnet fixtures" setup=[ParitySetup] begin
    for name in ["resnet_basic", "resnet_bottleneck", "resnext",
        "resnet_deep_avgdown", "resnet_tiered"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testitem "Models: timm resnet architectures" setup=[ParitySetup] begin
    for arch in list_models()
        startswith(arch, r"resne|wide_resnet") || continue
        @test check_param_counts(arch)
    end
end

@testitem "Models: timm resnet gradient" setup=[ParitySetup] begin
    using Flux
    m = Flux.trainmode!(create_model(fixture("resnet_basic"); pretrained = true))
    x = rand(Float32, 32, 32, 3, 2)
    g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
    @test all(isfinite, g.layer1.layers[1].conv1.weight)
    @test all(isfinite, g.conv1.weight)
end

@testitem "Models: timm resnet references" tags=[:network] setup=[ParitySetup] begin
    for file in reference_files("resnet")∪reference_files("resnext")∪reference_files("wide_resnet")
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true), ref["pre_logits"]) <=
              1e-4
    end
end
