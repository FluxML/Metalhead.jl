using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "GoogLeNet" begin
    @testset for bn in [true, false]
        m = GoogLeNet(; batchnorm = bn) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (GoogLeNet, bn) in PRETRAINED_MODELS
            @test acctest(GoogLeNet(; batchnorm = bn, pretrain = true))
        else
            @test_throws ArgumentError GoogLeNet(batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
