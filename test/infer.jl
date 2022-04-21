@testset "Inferrability" begin
    r = rand(Float32, 56, 56, 64, 1)
    @inferred gradient((x,y) -> sum(Metalhead.cat_channels(x, y)), r, r)
end
