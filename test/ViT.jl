using Metalhead, Test
using Flux

@testset "MLPMixer" begin
    @test size(MLPMixer()(rand(Float32, 256, 256, 3, 67))) == (1000, 67)
    @test_skip gradtest(MLPMixer(), rand(Float32, 256, 256, 3, 67))
end