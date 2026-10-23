using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["inception_v3", "inception_v4", "inception_resnet_v2"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in ["inception_v3", "inception_v4", "inception_resnet_v2"]
        @test arch in list_models()
        @test check_param_counts(arch)
    end
    @test_throws "aux_logits" create_model("inception_v3"; aux_logits = true)
end

@testset "gradient" begin
    using Flux, Functors
    for (name, in_chans) in ["inception_v3" => 1, "inception_resnet_v2" => 2]
        m = Flux.trainmode!(fixture_model(fixture(name); eltype = Float32))
        x = rand(Float32, 83, 83, in_chans, 2)
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        grads = filter(a -> a isa AbstractArray, Functors.fleaves(g))
        @test length(grads) == length(Flux.trainables(m))
        @test all(a -> all(isfinite, a), grads)
    end
end

@testset "forward_intermediates" begin
    using Flux
    m = fixture_model(fixture("inception_v4"); eltype = Float32)  # channels shrunk by 32
    x = rand(Float32, 75, 75, 1, 2)
    feats, ys = forward_intermediates(m, x)
    @test feats ≈ forward_features(m, x)
    @test [size(y, 3) for y in ys] == [f.num_chs for f in feature_info(m)] ==
          [2, 5, 12, 32, 48]
    @test forward_intermediates(m, x; indices = 2)[2] ≈ ys[4:5]
    @test forward_intermediates(m, x; indices = [4, 2], intermediates_only = true) ≈
          ys[[2, 4]]
    early, sel = forward_intermediates(m, x; indices = [3], stop_early = true)
    @test early ≈ sel[1] ≈ ys[3]
    @test_throws "indices" forward_intermediates(m, x; indices = [0])
    loss(m) = sum(forward_intermediates(m, x; indices = [2], intermediates_only = true)[1])
    g = Flux.gradient(loss, m)[1]
    @test all(isfinite, g.features.layers[1].layers.conv.weight)
    @test all(isfinite, g.features.layers[4].conv.layers.conv.weight)
    @test g.features.layers[5] === nothing
end

@testset "shrink_channels" begin
    using Flux
    m = create_model("resnet18"; num_classes = 10)
    s = shrink_channels(m, 8; keep = (3, 10))
    @test size(s.conv1.weight) == (7, 7, 3, 8)
    @test size(s.layer1.layers[1].conv1.weight) == (3, 3, 8, 8)
    @test s.bn1.chs == 8 && length(s.bn1.μ) == 8
    @test size(s.fc.weight) == (10, 64)
    @test size(s(rand(Float32, 64, 64, 3, 2))) == (10, 2)
    @test s.conv1.stride == m.conv1.stride && s.conv1.pad == m.conv1.pad

    x = create_model("resnext50_32x4d"; num_classes = 10, layers = (1, 1, 1, 1))
    @test_throws "not divisible" shrink_channels(x, 3; keep = (3, 10))
    @test size(shrink_channels(x, 4; keep = (3, 10)).layer1.layers[1].conv2.weight) ==
          (3, 3, 1, 32)
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("inception_v") ∪ reference_files("inception_resnet")
        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x = reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true),
            ref["pre_logits"]) <= 1e-4
    end
end
