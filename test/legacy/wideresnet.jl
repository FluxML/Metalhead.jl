using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "WideResNet" begin
    sizes = TEST_FAST ? [50] : [50, 101]
    @testset "WideResNet($sz)" for sz in sizes
        m = WideResNet(sz) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
        if (WideResNet, sz) in PRETRAINED_MODELS
            @test acctest(WideResNet(sz; pretrain = true))
        else
            @test_throws ArgumentError WideResNet(sz, pretrain = true)
        end
    end
end
