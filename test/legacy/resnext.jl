using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "ResNeXt" begin
    depths = TEST_FAST ? [50] : [50, 101, 152]
    cardinalities = TEST_FAST ? [32] : [32, 64]
    base_widths = TEST_FAST ? [4] : [4, 8]
    @testset for depth in depths, cardinality in cardinalities, base_width in base_widths
        m = ResNeXt(depth; cardinality, base_width) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (ResNeXt, depth, cardinality, base_width) in PRETRAINED_MODELS
            @test acctest(ResNeXt(depth; cardinality, base_width, pretrain = true))
        else
            @test_throws ArgumentError ResNeXt(depth;
                cardinality,
                base_width,
                pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end
