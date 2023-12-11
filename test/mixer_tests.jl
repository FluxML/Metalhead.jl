@testitem "MLP-Mixer" setup=[TestModels] begin
    @testset for config in [:small, :base, :large]
        m, x = MLPMixer(config), x_224
        @test size(m(x)) == (1000, 1)
        @test gradtest(m, x)
        _gc()
    end
end

@testitem "ResMLP" setup=[TestModels] begin
    @testset for config in [:small, :base, :large]
        m, x = ResMLP(config), x_224
        @test size(m(x)) == (1000, 1)
        @test gradtest(m, x)
        _gc()
    end
end

@testitem "gMLP" setup=[TestModels] begin
    @testset for config in [:small, :base, :large]
        m, x = gMLP(config), x_224
        @test size(m(x)) == (1000, 1)
        @test gradtest(m, x)
        _gc()
    end
end