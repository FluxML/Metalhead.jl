@testitem "ViT" setup=[TestModels] begin
    @testset for config in [:tiny, :small, :base, :large, :huge] # :giant, :gigantic]
        m = ViT(config)
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end
