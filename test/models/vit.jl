using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["vit_tiny", "vit_options", "vit_dinov2", "vit_dinov2_reg", "vit_dynamic",
        "vit_dynamic_reg", "vit_clip", "vit_siglip", "vit_siglip_gap", "vit_avgmax",
        "vit_max_nopos"]
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
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; global_pool = "prr")
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; act_layer = "relu")
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; dynamic_img_pad = true)
    @test_throws "class_token" create_model("vit_tiny_patch16_224"; class_token = false)
end

@testset "img_size from the pretrained cfg" begin
    dir = mktempdir()
    config = Dict("architecture" => "vit_small_patch14_dinov2", "num_classes" => 0,
        "model_args" => Dict("embed_dim" => 16, "depth" => 1, "num_heads" => 2),
        "pretrained_cfg" => Dict("input_size" => [3, 28, 42], "fixed_input_size" => true))
    write(joinpath(dir, "config.json"), JSON.json(config))
    @test create_model(dir; pretrained = false).grid_size == (2, 3)
    @test create_model(dir; pretrained = false, img_size = 14).grid_size == (1, 1)
end

@testset "forward_intermediates" begin
    m = fixture_model(fixture("vit_dinov2_reg"); eltype = Float32)
    x = rand(Float32, 28, 21, 3, 2)
    features, ys = forward_intermediates(m, x)
    @test features ≈ forward_features(m, x)
    @test size.(ys) == [(4, 3, 16, 2), (4, 3, 16, 2)]
    nlc = forward_intermediates(m, x; output_fmt = "NLC", intermediates_only = true)
    @test reshape(permutedims(ys[1], (3, 1, 2, 4)), 16, 12, 2) ≈ nlc[1]
    pairs = forward_intermediates(m, x; return_prefix_tokens = true, intermediates_only = true)
    @test first(pairs[2]) ≈ ys[2]
    @test size(last(pairs[2])) == (16, 3, 2)
    @test forward_intermediates(m, x; indices = 1, intermediates_only = true) ≈ ys[2:2]
    truncated, y1 = forward_intermediates(m, x; indices = [1], stop_early = true)
    @test y1 ≈ ys[1:1]
    @test truncated ≈ m.norm(m.blocks.layers[1](Metalhead.Timm._embed(m, x)))
    @test_throws "indices" forward_intermediates(m, x; indices = [3])
    @test_throws "output_fmt" forward_intermediates(m, x; output_fmt = "NHWC")
    @test feature_info(m) ==
          [(; num_chs = 16, reduction = 7, module_name = "blocks.$i") for i in 0:1]
    g = Metalhead.Flux.gradient(
        m -> sum(abs2, only(forward_intermediates(m, x; indices = [1],
            intermediates_only = true))), m)[1]
    @test all(isfinite, g.blocks.layers[1].attn.qkv.weight)
    @test g.blocks.layers[2] === nothing
end

@testset "dynamic image size" begin
    m = fixture_model(fixture("vit_dynamic"); eltype = Float32)
    for (w, h) in [(28, 28), (21, 35), (42, 14)]
        @test size(m(rand(Float32, w, h, 3, 1))) == (16, 1)
        @test size(only(forward_intermediates(m, rand(Float32, w, h, 3, 1);
            indices = 1, intermediates_only = true))) == (w ÷ 7, h ÷ 7, 16, 1)
    end
end

@testset "half precision" begin
    for name in ["vit_tiny", "vit_siglip"]
        @test check_half_precision(name)
    end
    m = fixture_model(fixture("vit_dynamic"); eltype = Float32)
    @test check_half_precision(m, rand(Float32, 21, 35, 3, 2))
    # half-precision position embeddings are resampled in Float32, as in timm
    posemb = Float16.(randn(Float32, 8, 1 + 4 * 4, 1))
    @test Metalhead.Timm.resample_abs_pos_embed(posemb, (3, 5)) ==
          Float16.(Metalhead.Timm.resample_abs_pos_embed(Float32.(posemb), (3, 5)))
end

@testset "gradient" begin
    for (name, x) in [("vit_options", rand(Float32, 32, 16, 2, 2)),
        ("vit_dinov2_reg", rand(Float32, 28, 21, 3, 2)),
        ("vit_dynamic", rand(Float32, 21, 35, 3, 2)),
        ("vit_clip", rand(Float32, 32, 32, 3, 2)),
        ("vit_siglip", rand(Float32, 32, 32, 3, 2))]
        @test check_gradient(fixture_model(fixture(name); eltype = Float32), x)
    end
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
