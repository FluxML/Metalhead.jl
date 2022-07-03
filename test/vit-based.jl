@testset "ViT" begin
    for mode in [:small, :base, :large] # :tiny, #,:huge, :giant, :gigantic]
        m = ViT(mode)
        @test size(m(x_256)) == (1000, 1)
        @test gradtest(m, x_256)
        _gc()
    end
end
