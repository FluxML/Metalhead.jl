using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "VGG" begin
    sizes = TEST_FAST ? [11] : [11, 13, 16, 19]
    @testset "VGG($sz, batchnorm=$bn)" for sz in sizes, bn in [true, false]
        m = VGG(sz; batchnorm = bn) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (VGG, sz, bn) in PRETRAINED_MODELS
            @test acctest(VGG(sz; batchnorm = bn, pretrain = true))
        else
            @test_throws ArgumentError VGG(sz, batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
