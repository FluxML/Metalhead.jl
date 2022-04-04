using Metalhead, Test
using Flux

@testset "ViT" begin
  for mode in [:tiny, :small, :base, :large] #,:huge, :giant, :gigantic]
    m = ViT(mode)
    @test size(m(rand(Float32, 224, 224, 3, 1))) == (1000, 1)
    @test_skip gradtest(m, rand(Float32, 224, 224, 3, 1))
  end
  GC.gc()
end
