using Metalhead, Test
using Flux

@testset "MLPMixer" begin
    @test size(MLPMixer()(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
    @test_skip gradtest(MLPMixer(), rand(Float32, 256, 256, 3, 2))
end