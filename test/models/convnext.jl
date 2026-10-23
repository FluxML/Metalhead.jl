using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["convnext_conv_mlp", "convnext_linear_mlp"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, "convnext") || continue
        @test check_param_counts(arch)
    end
end

@testset "gradient" begin
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

@testset "half precision" begin
    for name in ["convnext_conv_mlp", "convnext_linear_mlp"]
        @test check_half_precision(name)
    end
end

@testset "forward_intermediates" begin
    using Flux
    m = fixture_model(fixture("convnext_linear_mlp"); eltype = Float32)  # patch_size = 2
    x = rand(Float32, 64, 64, 3, 2)
    feats, ys = forward_intermediates(m, x)
    @test feats ≈ forward_features(m, x)
    @test [size(y, 3) for y in ys] == [f.num_chs for f in feature_info(m)]
    @test [64 ÷ size(y, 1) for y in ys] == [f.reduction for f in feature_info(m)]
    @test forward_intermediates(m, x; indices = 1, norm = true)[2] ≈ ys[4:4]
    @test forward_intermediates(m, x; indices = [2, 4], intermediates_only = true) ≈
          ys[[2, 4]]
    early, sel = forward_intermediates(m, x; indices = [3], stop_early = true)
    @test early ≈ sel[1] ≈ ys[3]
    @test_throws "indices" forward_intermediates(m, x; indices = [5])
    @test_throws "output_fmt" forward_intermediates(m, x; output_fmt = "NLC")
    loss(m) = sum(forward_intermediates(m, x; indices = [2], intermediates_only = true)[1])
    g = Flux.gradient(loss, m)[1]
    @test all(isfinite, g.stem.layers[1].weight)
    @test all(isfinite, g.stages.layers[2].blocks.layers[1].gamma)
    @test g.stages.layers[3] === nothing
end

TEST_NETWORK && @testset "references" begin
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
