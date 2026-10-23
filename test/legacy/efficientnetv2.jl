using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "EfficientNetv2" begin
    configs = TEST_FAST ? [:small] : [:small, :medium, :large] # :xlarge]
    @testset for config in configs
        m = EfficientNetv2(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (EfficientNetv2, config) in PRETRAINED_MODELS
            @test acctest(EfficientNetv2(config; pretrain = true))
        else
            @test_throws ArgumentError EfficientNetv2(config, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
