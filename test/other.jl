using Metalhead, Test
using Flux

@testset "MLPMixer" begin
  @testset for mode in [:small, :base, :large, :huge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = MLPMixer(mode; drop_path_rate)
      @test size(m(rand(Float32, 224, 224, 3, 1))) == (1000, 1)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 1))
    end
    GC.gc()
  end
end

@testset "ResMLP" begin
  @testset for mode in [:small, :base, :large, :huge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = ResMLP(mode; drop_path_rate)
      @test size(m(rand(Float32, 224, 224, 3, 1))) == (1000, 1)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 1))
    end
    GC.gc()
  end
end

@testset "gMLP" begin
  @testset for mode in [:small, :base, :large, :huge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = gMLP(mode; drop_path_rate)
      @test size(m(rand(Float32, 224, 224, 3, 1))) == (1000, 1)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 1))
    end
    GC.gc()
  end
end
