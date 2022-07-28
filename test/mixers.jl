@testset "MLPMixer" begin
	@testset for mode in [:small, :base, :large, :huge]
		@testset for drop_path_rate in [0.0, 0.5]
			m = MLPMixer(mode; drop_path_rate)
			@test size(m(x_224)) == (1000, 1)
			@test gradtest(m, x_224)
			_gc()
		end
	end
end

@testset "ResMLP" begin
    @testset for mode in [:small, :base, :large, :huge]
        @testset for drop_path_rate in [0.0, 0.5]
            m = ResMLP(mode; drop_path_rate)
            @test size(m(x_224)) == (1000, 1)
            @test gradtest(m, x_224)
            _gc()
        end
    end
end

@testset "gMLP" begin
    @testset for mode in [:small, :base, :large, :huge]
		@testset for drop_path_rate in [0.0, 0.5]
			m = gMLP(mode; drop_path_rate)
			@test size(m(x_224)) == (1000, 1)
			@test gradtest(m, x_224)
			_gc()
		end
    end
end
