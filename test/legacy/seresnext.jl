using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "SEResNeXt" begin
    depths = TEST_FAST ? [50] : [50, 101, 152]
    cardinalities = TEST_FAST ? [32] : [32, 64]
    base_widths = TEST_FAST ? [4] : [4, 8]
    @testset for depth in depths, cardinality in cardinalities, base_width in base_widths
        m = SEResNeXt(depth; cardinality, base_width) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (SEResNeXt, depth, cardinality, base_width) in PRETRAINED_MODELS
            @test acctest(SEResNeXt(depth; pretrain = true))
        else
            @test_throws ArgumentError SEResNeXt(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
