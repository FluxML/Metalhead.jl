@testset for model in [MLPMixer, ResMLP, gMLP]
    @testset for config in [:small, :base, :large]
        m = model(config)
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end
