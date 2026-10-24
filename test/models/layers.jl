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

@testset "mixed-precision norms" begin
    using Flux
    using Metalhead.Timm: LayerNorm2d
    using Metalhead.Transformers: RMSNorm
    for (norm, x) in [(RMSNorm(4; eps = 1e-6), randn(Float32, 4, 3, 2)),
        (LayerNorm2d(4), randn(Float32, 5, 3, 4, 2))]
        foreach(p -> p .= randn(Float32, size(p)), Flux.trainables(norm))
        half, xh = f16(norm), f16(x)
        @test Flux.trainables(half) == Flux.trainables(norm)
        @test all(p -> p isa Vector{Float32}, Flux.trainables(half))
        @test all(p -> p isa Vector{Float32}, Flux.trainables(bf16(norm)))
        @test all(p -> p isa Vector{Float32}, Flux.trainables(cpu(half)))
        @test half(xh) isa Array{Float16}
        @test half(xh) ≈ f16(norm(Float32.(xh))) rtol=1e-3
        @test f64(norm)(Float64.(x)) isa Array{Float64}
        g = Flux.gradient(m -> sum(f16(m)(xh)), norm)[1]
        @test g.weight isa Vector{Float32}
    end
end
