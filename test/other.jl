using Metalhead, Test
using Flux


@testset "MLPMixer" begin
    @test size(MLPMixer()(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
    @test_skip gradtest(MLPMixer(), rand(Float32, 256, 256, 3, 2))
end

@testset "ESRGAN" begin
    esrgan = ESRGAN()
    D = esrgan[:discrimator]
    G = esrgan[:generator]
    @test size(G(rand(Float32,24,24,3,5))) == (96, 96, 3, 5)
    @test size(D(G(rand(Float32,24,24,3,5)))) == (1,5)
end