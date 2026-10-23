using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["vgg_plain", "vgg_bn_adaptive_h", "vgg_adaptive_w"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "feature extraction" begin
    for name in ["vgg_plain", "vgg_bn_adaptive_h"]
        check_intermediate_api(name)
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, "vgg") || continue
        @test check_param_counts(arch)
    end
end

@testset "gradient" begin
    using Flux
    for (name, x) in [("vgg_bn_adaptive_h", rand(Float32, 64, 40, 2, 2)),
        ("vgg_adaptive_w", rand(Float32, 20, 64, 3, 2))]
        m = Flux.trainmode!(create_model(fixture(name); pretrained = true))
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        @test all(isfinite, g.features.layers[1].weight)
        w = g.pre_logits.fc1.weight
        @test all(isfinite, w) && !all(iszero, w)
        @test all(isfinite, g.head.fc.weight)
    end
end

@testset "options" begin
    using Metalhead.Timm: adaptive_avg_pool2d, ClassifierHead
    using Flux

    # the bn variants add a BatchNorm after each convolution
    tiny = (; cfg = (4, "M", 8), mlp_ratio = 1 / 512, num_classes = 3)
    @test length(create_model("vgg11"; tiny...).features.layers) == 5
    @test create_model("vgg11_bn"; tiny...).features.layers[2] isa BatchNorm
    @test create_model("vgg11"; tiny..., norm_layer = "batchnorm2d").features.layers[2] isa
          BatchNorm
    headless = create_model("vgg13"; tiny..., num_classes = 0)
    @test size(headless(rand(Float32, 8, 8, 3, 1))) == (4096, 1)
    @test_throws "not supported" create_model("vgg11"; norm_layer = "layernorm2d")
    @test_throws "not supported" create_model("vgg11"; global_pool = "max")
    @test_throws "not supported" create_model("vgg11"; act_layer = "gelu")
    @test_throws "not supported" create_model("vgg11"; output_stride = 16)

    # adaptive pooling is the identity for equal sizes, and replicates a single pixel
    x = rand(Float32, 3, 2, 2, 1)
    @test adaptive_avg_pool2d(x, (3, 2)) == x
    @test adaptive_avg_pool2d(x[1:1, 1:1, :, :], (7, 7)) == repeat(x[1:1, 1:1, :, :], 7, 7)
    @test size(adaptive_avg_pool2d(x, (7, 2))) == (7, 2, 2, 1)

    head = ClassifierHead(8, 5)
    feats = rand(Float32, 3, 3, 8, 2)
    @test size(head(feats)) == (5, 2)
    pooled = dropdims(sum(feats; dims = (1, 2)) / 9; dims = (1, 2))
    @test head(feats; pre_logits = true) ≈ pooled
    @test ClassifierHead(8, 0)(feats) == head(feats; pre_logits = true)
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("vgg")
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        pre_logits = forward_head(model, forward_features(model, x); pre_logits = true)
        @test relerr(pre_logits, ref["pre_logits"]) <= 1e-4
    end
end
