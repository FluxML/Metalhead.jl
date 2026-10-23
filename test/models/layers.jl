using Metalhead, Test

@testset "timm layers" begin
    using Metalhead: torch_params
    using Metalhead.Timm: create_conv2d, conv_norm_act, make_divisible, Conv2dSame
    using Flux

    @test [make_divisible(v, d) for (v, d) in
           [(16 * 0.25, 8), (37.5, 8), (92.16, 8), (3.3, 8), (1408, 8), (13.5, 4)]] ==
          [8, 40, 96, 8, 1408, 16]
    @test make_divisible(37.5, 8; round_limit = 0.0) == 40
    @test make_divisible(4.4, 1; min_value = 16) == 16

    # TF "SAME" padding puts the odd pixel last; static cases are plain Convs
    same = create_conv2d(4, 8, 3; stride = 2, padding = "same")
    @test same isa Conv2dSame
    @test create_conv2d(4, 8, 5; padding = "same").pad == (2, 2)
    @test create_conv2d(4, 8, (1, 7); padding = (0, 3)).pad == (3, 0)
    for (w, h, pad) in [(7, 9, (1, 1, 1, 1)), (8, 6, (0, 1, 0, 1))]
        x = rand(Float32, w, h, 4, 2)
        padded = Conv(same.conv.weight, false; stride = 2, pad)
        @test same(x) ≈ padded(x)
        @test size(same(x))[1:2] == cld.((w, h), 2)
    end
    @test [p.key for p in first(torch_params((conv_stem = same,)))] == ["conv_stem.weight"]
    @test [p.key for p in first(torch_params(conv_norm_act(4, 8, 3)))] ==
          ["conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]
end
