using Metalhead, Test
using Flux

@testset "ViT" begin
    @test size(ViT()(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
    @test_skip gradtest(ViT(), rand(Float32, 256, 256, 3, 2))
end