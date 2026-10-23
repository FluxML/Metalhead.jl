using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "AlexNet" begin
    model = AlexNet() |> gpu
    @test size(model(x_256)) == (1000, 1)
    @test_throws ArgumentError AlexNet(pretrain = true)
    @test gradtest(model, x_256)
    _gc()
end
