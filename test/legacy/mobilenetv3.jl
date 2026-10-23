using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "MobileNetv3" begin
    configs = TEST_FAST ? [:small] : [:small, :large]
    @testset for width_mult in WIDTH_MULTS, config in configs
        m = MobileNetv3(config; width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv3, config, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv3(config; pretrain = true))
        else
            @test_throws ArgumentError MobileNetv3(config; pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
