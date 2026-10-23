using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "MobileNetV1" begin
    @testset for width_mult in WIDTH_MULTS
        m = MobileNetv1(width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv1, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv1(; pretrain = true))
        else
            @test_throws ArgumentError MobileNetv1(pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
