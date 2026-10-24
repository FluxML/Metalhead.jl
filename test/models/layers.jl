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

@testset "attention dropout" begin
    using Metalhead.Timm: attention_fdrop
    using Flux
    # NNlib takes its fused attention only with `fdrop = identity`, so an inactive dropout
    # is not passed on
    d, x = Dropout(0.1), rand(Float32, 4, 2, 3, 1)
    @test attention_fdrop(d, x) === identity
    @test attention_fdrop(identity, x) === identity
    @test attention_fdrop(Dropout(0.0), x) === identity
    active = Ref{Any}()
    Flux.gradient(x -> (active[] = attention_fdrop(d, x); sum(x)), x)
    @test active[] === d
    @test attention_fdrop(Flux.trainmode!(Dropout(0.1)), x) isa Dropout
    Flux.testmode!(d)
    Flux.gradient(x -> (active[] = attention_fdrop(d, x); sum(x)), x)
    @test active[] === identity
    # the attention layers pass the dropout of their model only in training
    model = create_model("LlamaForCausalLM"; vocab_size = 16, hidden_size = 8,
        intermediate_size = 12, num_hidden_layers = 1, num_attention_heads = 2,
        attention_dropout = 0.5)
    ids = rand(1:16, 5, 2)
    @test model(ids) == model(ids)
    Flux.trainmode!(model)
    @test model(ids) != model(ids)
end

@testset "mixed-precision norms" begin
    using Flux
    using Metalhead.Timm: LayerNorm2d
    using Metalhead.Transformers: RMSNorm
    for (norm, x) in [(RMSNorm(4; eps = 1e-6), randn(Float32, 4, 3, 2)),
        (LayerNorm2d(4), randn(Float32, 5, 3, 4, 2))]
        foreach(p -> p .= randn(Float32, size(p)), Flux.trainables(norm))
        half, xh = f16(norm), f16(x)
        @test all(p -> eltype(p) == Float32, Flux.trainables(half))
        @test half(xh) isa Array{Float16}
        @test half(xh) ≈ f16(norm(Float32.(xh))) rtol=1e-3
        @test f64(norm)(Float64.(x)) isa Array{Float64}
    end
end
