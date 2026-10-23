using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["densenet_basic", "densenet_blur", "densenet_tiered",
        "densenet_tiered_narrow"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "feature extraction" begin
    for name in ["densenet_basic", "densenet_blur"]
        check_intermediate_api(name)
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, "densenet") || continue
        @test check_param_counts(arch)
    end
end

@testset "gradient" begin
    using Flux
    for (name, x) in [("densenet_basic", rand(Float32, 48, 64, 3, 2)),
        ("densenet_blur", rand(Float32, 64, 64, 2, 2))]
        m = Flux.trainmode!(create_model(fixture(name); pretrained = true))
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        block = g.features.layers.denseblock1.layers.denselayer1.layers
        @test all(isfinite, g.features.layers.conv0.weight)
        @test all(isfinite, block.conv1.weight) && !all(iszero, block.conv1.weight)
        @test all(isfinite, g.features.layers.transition1.layers.conv.weight)
        @test all(isfinite, g.classifier.weight)
    end
end

@testset "options" begin
    using Metalhead.Timm: BlurPool2d
    using Metalhead: torch_params
    using Flux

    # blur pooling has no parameters and preserves constants
    @test isempty(first(torch_params(BlurPool2d(3))))
    @test BlurPool2d(3)(fill(2.0f0, 8, 8, 3, 1)) ≈ fill(2.0f0, 4, 4, 3, 1)
    @test size(BlurPool2d()(rand(Float32, 9, 7, 5, 2))) == (5, 4, 5, 2)

    tiny = (; growth_rate = 4, block_config = (1, 1), num_classes = 3)
    features(; kw...) = create_model("densenet121"; tiny..., kw...).features.layers
    @test features().pool0 isa MaxPool
    @test features(aa_layer = "blurpool2d").pool0 isa Chain
    @test features(aa_layer = "blurpool2d").transition1.layers.pool isa MeanPool
    blur_all = features(aa_layer = "blurpool2d", aa_stem_only = false)
    @test blur_all.transition1.layers.pool isa BlurPool2d
    @test_throws "not supported" create_model("densenet121"; aa_layer = "blur")
    @test_throws "not supported" create_model("densenet121"; norm_layer = "layernorm2d")
    @test_throws "not supported" create_model("densenet121"; global_pool = "max")
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("densenet")
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        pre_logits = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(pre_logits, ref["pre_logits"]) <= 1e-4
    end
end
