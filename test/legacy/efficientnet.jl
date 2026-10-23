using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "EfficientNet" begin
    configs = TEST_FAST ? [:b0] : [:b0, :b1, :b2, :b3, :b4, :b5] #:b6, :b7, :b8]
    @testset "EfficientNet($config)" for config in configs
        # preferred image resolution scaling
        r = Metalhead.EFFICIENTNET_GLOBAL_CONFIGS[config][1]
        x = rand(Float32, r, r, 3, 1) |> gpu
        m = EfficientNet(config) |> gpu
        @test size(m(x)) == (1000, 1)
        if (EfficientNet, config) in PRETRAINED_MODELS
            @test acctest(EfficientNet(config; pretrain = true))
        else
            @test_throws ArgumentError EfficientNet(config, pretrain = true)
        end
        @test gradtest(m, x)
        _gc()
    end
end
