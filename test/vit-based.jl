using Metalhead, Test
using Flux

@testset "ViT" begin
	for mode in [:tiny, :small, :base, :large, :huge, :giant, :gigantic]
		m = ViT(mode)
		@test size(m(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
		@test_skip gradtest(m, rand(Float32, 256, 256, 3, 2))
		GC.gc()
	end
end