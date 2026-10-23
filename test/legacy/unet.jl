using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "UNet" begin
    encoder = Metalhead.backbone(ResNet(18))
    model = UNet((256, 256), 3, 10, encoder) |> gpu
    @test size(model(x_256)) == (256, 256, 10, 1)
    @test gradtest(model, x_256)

    model = UNet() |> gpu
    @test size(model(x_256)) == (256, 256, 3, 1)
    _gc()
end
