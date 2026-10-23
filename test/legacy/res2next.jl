using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "Res2NeXt" begin
    @testset for depth in [50, 101]
        m = Res2NeXt(depth) |> gpu
        if has_cuda()
            @test_broken size(m(x_224)) == (1000, 1)
        else
            @test size(m(x_224)) == (1000, 1)
        end
        if (Res2NeXt, depth) in PRETRAINED_MODELS
            @test acctest(Res2NeXt(depth; pretrain = true))
        else
            @test_throws ArgumentError Res2NeXt(depth, pretrain = true)
        end
        if has_cuda()
            @test_broken gradtest(m, x_224)
        else
            @test gradtest(m, x_224)
        end
        _gc()
    end
end
