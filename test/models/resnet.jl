using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["resnet_basic", "resnet_bottleneck", "resnext",
        "resnet_deep_avgdown", "resnet_tiered", "senet", "seresnet_basic", "ecaresnet"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, r"resne|wide_resnet|seresne|senet|ecaresne") || continue
        @test check_param_counts(arch)
    end
end

@testset "gradient" begin
    x = rand(Float32, 32, 32, 3, 2)
    for name in ["resnet_basic", "senet", "seresnet_basic", "ecaresnet"]
        @test check_gradient(fixture_model(fixture(name); eltype = Float32), x)
    end
end

@testset "attention layers" begin
    using Metalhead.Timm: SEModule, EcaModule, create_attn
    using Metalhead: create_model

    # sizes of timm.layers.SEModule(c) and EcaModule(c)
    for (c, rd) in [(16, 8), (64, 8), (192, 16), (256, 16), (512, 32), (2048, 128)]
        @test size(SEModule(c).fc1.weight) == (1, 1, c, rd)
    end
    @test size(SEModule(256; rd_ratio = 0.25).fc1.weight, 4) == 64
    for (c, k) in [(16, 3), (64, 3), (256, 5), (512, 5), (1024, 5), (2048, 7), (4096, 7)]
        @test size(EcaModule(c).conv.weight) == (k, 1, 1)
    end
    @test create_attn(nothing, 8) === nothing
    @test create_attn("SE", 8) isa SEModule
    @test_throws "not supported" create_attn("cbam", 8)

    # `block_args` arrives as a NamedTuple from config.json and is validated
    tiny = (; layers = (1, 1, 1, 1), channels = (8, 16, 32, 64), num_classes = 10)
    se(; kw...) = create_model("resnet18"; tiny..., kw...).layer1.layers[1].se
    @test se(block_args = (attn_layer = "se",)) isa SEModule
    @test se(block_args = Dict("attn_layer" => "eca")) isa EcaModule
    @test se(block_args = (attn_layer = nothing,)) === nothing
    @test_throws "block_args" se(block_args = (scale = 4,))
    @test_throws "not supported" se(block_args = (attn_layer = "cbam",))
    @test_throws "block_args" create_model("res2net50_26w_4s"; tiny...,
        block_args = (attn_layer = "se",))
end

TEST_NETWORK && @testset "references" begin
    prefixes = ["resnet", "resnext", "wide_resnet", "seresne", "ecaresne"]
    for file in reduce(∪, reference_files.(prefixes))
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true), ref["pre_logits"]) <=
              1e-4
    end
end
