using Metalhead, Test
include("setup.jl")
using .TestModels

@testset "ResNet" begin
    # Tests for pretrained ResNets
    sizes = TEST_FAST ? [18] : [18, 34, 50, 101, 152]
    @testset "ResNet($sz)" for sz in sizes
        m = ResNet(sz) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (ResNet, sz) in PRETRAINED_MODELS
            @test acctest(ResNet(sz; pretrain = true))
        else
            @test_throws ArgumentError ResNet(sz, pretrain = true)
        end
    end

    @testset "resnet" begin
        @testset for block_fn in [Metalhead.basicblock, Metalhead.bottleneck]
            layer_list = [
                [2, 2, 2, 2],
                [3, 4, 6, 3],
                [3, 4, 23, 3],
                [3, 8, 36, 3],
            ]
            @testset for layers in layer_list
                drop_list = [
                    (dropout_prob = 0.1, stochastic_depth_prob = 0.1, dropblock_prob = 0.1),
                    (dropout_prob = 0.5, stochastic_depth_prob = 0.5, dropblock_prob = 0.5),
                    (dropout_prob = 0.8, stochastic_depth_prob = 0.8, dropblock_prob = 0.8),
                ]
                @testset for drop_probs in drop_list
                    m = Metalhead.resnet(block_fn, layers; drop_probs...) |> gpu
                    @test size(m(x_224)) == (1000, 1)
                    @test gradtest(m, x_224)
                    _gc()
                end
            end
        end
    end
end
