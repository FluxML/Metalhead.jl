using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "DenseNet" begin
    sizes = TEST_FAST ? [121] : [121, 161, 169, 201]
    @testset for sz in sizes
        m = DenseNet(sz) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (DenseNet, sz) in PRETRAINED_MODELS
            @test acctest(DenseNet(sz; pretrain = true))
        else
            @test_throws ArgumentError DenseNet(sz, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
