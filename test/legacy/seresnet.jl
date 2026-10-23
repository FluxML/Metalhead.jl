using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "SEResNet" begin
    depths = TEST_FAST ? [18] : [18, 34, 50, 101, 152]
    @testset for depth in depths
        m = SEResNet(depth) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (SEResNet, depth) in PRETRAINED_MODELS
            @test acctest(SEResNet(depth; pretrain = true))
        else
            @test_throws ArgumentError SEResNet(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
