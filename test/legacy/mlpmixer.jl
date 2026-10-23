using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "MLP-Mixer" begin
    configs = TEST_FAST ? [:small] : [:small, :base, :large]
    @testset for config in configs
        m = MLPMixer(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end
