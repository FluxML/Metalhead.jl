@testitem "ViT" setup=[TestModels] begin
    configs = TEST_FAST ? [:tiny] : [:tiny, :small, :base, :large, :huge] # :giant, :gigantic]
    @testset for config in configs
        m = ViT(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if VERSION < v"1.7" && has_cuda()
            @test_broken gradtest(m, x_224)
        else
            @test gradtest(m, x_224)
        end
        _gc()
    end
end
