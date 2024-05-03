@testitem "ViT" setup=[TestModels] begin
    configs = TEST_FAST ? [:tiny] : [:tiny, :small, :base, :large, :huge] # :giant, :gigantic]
    @testset for config in configs
        m = ViT(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end
