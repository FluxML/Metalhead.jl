using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "Inception" begin
    x_299 = rand(Float32, 299, 299, 3, 2) |> gpu
    @testset "$Model" for Model in [Inceptionv3, Inceptionv4, InceptionResNetv2, Xception]
        m = Model() |> gpu
        @test size(m(x_299)) == (1000, 2)
        if Model in PRETRAINED_MODELS
            @test acctest(Model(; pretrain = true))
        else
            @test_throws ArgumentError Model(pretrain = true)
        end
        @test gradtest(m, x_299)
        _gc()
    end
end
