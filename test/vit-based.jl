using Metalhead, Test
using Flux

@testset "ViT" begin
    @test size(ViT()(rand(Float32, 256, 256, 3, 67))) == (1000, 67)
    @test_skip gradtest(ViT(), rand(Float32, 256, 256, 3, 67))
end