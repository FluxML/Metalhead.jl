using Metalhead, Test
using Flux

@testset "ViT" begin
  for mode in [:tiny, :small, :base, :large] #,:huge, :giant, :gigantic]
    m = ViT(mode)
    @test size(m(x_256)) == (1000, 1)
    @test gradtest(m, x_256)
    GC.gc()
  end
end
