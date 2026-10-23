using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "Res2Net" begin
    configs = TEST_FAST ? [(26, 4)] : [(26, 4), (48, 2), (14, 8), (26, 6), (26, 8)]
    @testset for (base_width, scale) in configs
        m = Res2Net(50; base_width, scale) |> gpu
        if has_cuda()
            @test_broken size(m(x_224)) == (1000, 1)
        else
            @test size(m(x_224)) == (1000, 1)
        end
        if (Res2Net, 50, base_width, scale) in PRETRAINED_MODELS
            if has_cuda()
                @test_broken acctest(Res2Net(50; base_width, scale, pretrain = true))
            else
                @test acctest(Res2Net(50; base_width, scale, pretrain = true))
            end
        else
            @test_throws ArgumentError Res2Net(50; base_width, scale, pretrain = true)
        end
        if has_cuda()
            @test_broken gradtest(m, x_224)
        else
            @test gradtest(m, x_224)
        end
        _gc()
    end

    if !TEST_FAST
        @testset for (base_width, scale) in [(26, 4)]
            m = Res2Net(101; base_width, scale) |> gpu
            @test size(m(x_224)) == (1000, 1)
            if (Res2Net, 101, base_width, scale) in PRETRAINED_MODELS
                @test acctest(Res2Net(101; base_width, scale, pretrain = true))
            else
                @test_throws ArgumentError Res2Net(101; base_width, scale, pretrain = true)
            end
            @test gradtest(m, x_224)
            _gc()
        end
    end
end
