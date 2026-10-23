using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "MobileNetv2" begin
    @testset for width_mult in WIDTH_MULTS
        m = MobileNetv2(width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv2, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv2(; pretrain = true))
        else
            @test_throws ArgumentError MobileNetv2(pretrain = true)
        end
        @test gradtest(m, x_224)
    end
end
