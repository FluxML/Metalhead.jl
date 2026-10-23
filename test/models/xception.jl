using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in
        ["xception_aligned", "xception_aligned_dilated", "xception41p", "xception65p"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in ["xception41", "xception65", "xception71", "xception41p", "xception65p"]
        @test arch in list_models()
        @test check_param_counts(arch)
    end
    @test_throws "act_layer" create_model("xception41"; act_layer = "gelu")
end

@testset "gradient" begin
    using Flux
    for (name, in_chans) in ["xception_aligned" => 2, "xception41p" => 3]
        m = Flux.trainmode!(fixture_model(fixture(name); eltype = Float32))
        x = rand(Float32, 35, 35, in_chans, 2)
        g = Flux.gradient(m -> sum(abs2, m(x)), m)[1]
        @test all(isfinite, g.stem.layers[1].layers.conv.weight)
        @test all(isfinite, g.head.fc.weight)
        @test all(isfinite, g.blocks.layers[end].stack.layers.conv3.conv_pw.weight)
    end
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("xception")
        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x = reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true),
            ref["pre_logits"]) <= 1e-4
    end
end
