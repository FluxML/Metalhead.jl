using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "SqueezeNet" begin
    m = SqueezeNet() |> gpu
    @test size(m(x_224)) == (1000, 1)
    if SqueezeNet in PRETRAINED_MODELS
        @test acctest(SqueezeNet(; pretrain = true))
    else
        @test_throws ArgumentError SqueezeNet(pretrain = true)
    end
    @test gradtest(m, x_224)
    _gc()
end
