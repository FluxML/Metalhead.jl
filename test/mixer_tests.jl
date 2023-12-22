@testitem "MLP-Mixer" setup=[TestModels] begin
    configs = TEST_FAST ? [:small] : [:small, :base, :large]
    @testset for config in configs
        m = MLPMixer(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "ResMLP" setup=[TestModels] begin
    configs = TEST_FAST ? [:small] : [:small, :base, :large]
    @testset for config in configs
        m = ResMLP(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "gMLP" setup=[TestModels] begin
    configs = TEST_FAST ? [:small] : [:small, :base, :large]
    @testset for config in configs
        m = gMLP(config) |> gpu
        if has_cuda()
            @test_broken size(m(x_224)) == (1000, 1)
            @test_broken gradtest(m, x_224)
        else
            @test size(m(x_224)) == (1000, 1)
            @test gradtest(m, x_224)
        end
        _gc()
    end
end