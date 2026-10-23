using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "MNASNet" begin
    @testset for width_mult in WIDTH_MULTS, config in [:A1, :B1]
        m = MNASNet(config; width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MNASNet, config, width_mult) in PRETRAINED_MODELS
            @test acctest(MNASNet(config; pretrain = true))
        else
            @test_throws ArgumentError MNASNet(config; pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
