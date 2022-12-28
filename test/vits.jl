@testset "ViT" begin for config in [:tiny, :small, :base, :large, :huge] # :giant, :gigantic]
    m = ViT(config)
    @test size(m(x_256)) == (1000, 1)
    @test gradtest(m, x_256)
    _gc()
end end
