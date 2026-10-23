using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["vit_tiny", "vit_options"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, "vit_") || continue
        @test check_param_counts(arch)
    end
end

@testset "unsupported options" begin
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; qk_norm = true)
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; init_values = 1e-5)
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; global_pool = "avg")
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; reg_tokens = 4)
end

@testset "gradient" begin
    using Flux
    m = Flux.trainmode!(create_model(fixture("vit_options"); pretrained = true))
    x = rand(Float32, 32, 16, 2, 2)
    g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
    @test all(isfinite, g.cls_token)
    @test all(isfinite, g.pos_embed)
    @test all(isfinite, g.patch_embed.proj.weight)
    @test all(isfinite, g.blocks.layers[1].attn.qkv.weight)
    @test all(isfinite, g.head.weight)
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("vit_")
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true), ref["pre_logits"]) <=
              1e-4
    end
end
