@testitem "AlexNet" setup=[TestModels] begin
    model = AlexNet() |> gpu
    @test size(model(x_256)) == (1000, 1)
    @test_throws ArgumentError AlexNet(pretrain = true)
    @test gradtest(model, x_256)
    _gc()
end

@testitem "VGG" setup=[TestModels] begin
    sizes = TEST_FAST ? [11] : [11, 13, 16, 19]
    @testset "VGG($sz, batchnorm=$bn)" for sz in sizes, bn in [true, false]
        m = VGG(sz; batchnorm = bn) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (VGG, sz, bn) in PRETRAINED_MODELS
            @test acctest(VGG(sz; batchnorm = bn, pretrain = true))
        else
            @test_throws ArgumentError VGG(sz, batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "ResNet" setup=[TestModels] begin
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
                    m = Metalhead.resnet(block_fn, layers; drop_probs...)
                    @test size(m(x_224)) == (1000, 1)
                    @test gradtest(m, x_224)
                    _gc()
                end
            end
        end
    end
end

@testitem "WideResNet" setup=[TestModels] begin
    sizes = TEST_FAST ? [50] : [50, 101]
    @testset "WideResNet($sz)" for sz in sizes
        m = WideResNet(sz) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
        if (WideResNet, sz) in PRETRAINED_MODELS
            @test acctest(WideResNet(sz; pretrain = true))
        else
            @test_throws ArgumentError WideResNet(sz, pretrain = true)
        end
    end
end

@testitem "ResNeXt" setup=[TestModels] begin
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

@testitem "SEResNet" setup=[TestModels] begin
    depths = TEST_FAST ? [18] : [18, 34, 50, 101, 152]
    @testset for depth in depths
        m = SEResNet(depth) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (SEResNet, depth) in PRETRAINED_MODELS
            @test acctest(SEResNet(depth; pretrain = true))
        else
            @test_throws ArgumentError SEResNet(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "SEResNeXt" setup=[TestModels] begin
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

@testitem "Res2Net" setup=[TestModels] begin
    configs = TEST_FAST ? [(26, 4)] : [(26, 4), (48, 2), (14, 8), (26, 6), (26, 8)]
    @testset for (base_width, scale) in configs
        m = Res2Net(50; base_width, scale) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (Res2Net, 50, base_width, scale) in PRETRAINED_MODELS
            @test acctest(Res2Net(50; base_width, scale, pretrain = true))
        else
            @test_throws ArgumentError Res2Net(50; base_width, scale, pretrain = true)
        end
        @test gradtest(m, x_224)
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

@testitem "Res2NeXt" setup=[TestModels] begin
    @testset for depth in [50, 101]
        m = Res2NeXt(depth) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (Res2NeXt, depth) in PRETRAINED_MODELS
            @test acctest(Res2NeXt(depth; pretrain = true))
        else
            @test_throws ArgumentError Res2NeXt(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "EfficientNet" setup=[TestModels] begin
    configs = TEST_FAST ? [:b0] : [:b0, :b1, :b2, :b3, :b4, :b5] #:b6, :b7, :b8]
    @testset "EfficientNet($config)" for config in configs
        # preferred image resolution scaling
        r = Metalhead.EFFICIENTNET_GLOBAL_CONFIGS[config][1]
        x = rand(Float32, r, r, 3, 1) |> gpu
        m = EfficientNet(config) |> gpu
        @test size(m(x)) == (1000, 1)
        if (EfficientNet, config) in PRETRAINED_MODELS
            @test acctest(EfficientNet(config; pretrain = true))
        else
            @test_throws ArgumentError EfficientNet(config, pretrain = true)
        end
        @test gradtest(m, x)
        _gc()
    end
end

@testitem "EfficientNetv2" setup=[TestModels] begin
    configs = TEST_FAST ? [:small] : [:small, :medium, :large] # :xlarge]
    @testset for config in configs
        m = EfficientNetv2(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (EfficientNetv2, config) in PRETRAINED_MODELS
            @test acctest(EfficientNetv2(config; pretrain = true))
        else
            @test_throws ArgumentError EfficientNetv2(config, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "GoogLeNet" setup=[TestModels] begin
    @testset for bn in [true, false]
        m = GoogLeNet(; batchnorm = bn) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (GoogLeNet, bn) in PRETRAINED_MODELS
            @test acctest(GoogLeNet(; batchnorm = bn, pretrain = true))
        else
            @test_throws ArgumentError GoogLeNet(batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "Inception" setup=[TestModels] begin
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

@testitem "SqueezeNet" setup=[TestModels] begin
    m = SqueezeNet() |> gpu
    @test size(m(x_224)) == (1000, 1)
    if SqueezeNet in PRETRAINED_MODELS
        if VERSION >= v"1.7"
            @test acctest(SqueezeNet(; pretrain = true))
        else
            @test_broken acctest(SqueezeNet(; pretrain = true))
        end
    else
        @test_throws ArgumentError SqueezeNet(pretrain = true)
    end
    @test gradtest(m, x_224)
    _gc()
end

@testitem "DenseNet" setup=[TestModels] begin
    sizes = TEST_FAST ? [121] : [121, 161, 169, 201]
    @testset for sz in sizes
        m = DenseNet(sz) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (DenseNet, sz) in PRETRAINED_MODELS
            @test acctest(DenseNet(sz; pretrain = true))
        else
            @test_throws ArgumentError DenseNet(sz, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testsetup module TestMobileNets
export WIDTH_MULTS
const WIDTH_MULTS = get(ENV, "FAST_TEST", "false") == "true" ? [0.5] : [0.5, 0.75, 1.0, 1.3]
end

@testitem "MobileNetV1" setup=[TestModels, TestMobileNets] begin
    @testset for width_mult in WIDTH_MULTS
        m = MobileNetv1(width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv1, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv1(; pretrain = true))
        else
            @test_throws ArgumentError MobileNetv1(pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "MobileNetv2" setup=[TestModels, TestMobileNets] begin
    @testset for width_mult in WIDTH_MULTS
        m = MobileNetv2(width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv2, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv2(; pretrain = true))
        else
            @test_throws ArgumentError MobileNetv2(pretrain = true)
        end
        @test gradtest(m, x_224)
    end
end

@testitem "MobileNetv3" setup=[TestModels, TestMobileNets] begin
    configs = TEST_FAST ? [:small] : [:small, :large]
    @testset for width_mult in WIDTH_MULTS, config in configs
        m = MobileNetv3(config; width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv3, config, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv3(config; pretrain = true))
        else
            @test_throws ArgumentError MobileNetv3(config; pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "MNASNet" setup=[TestModels, TestMobileNets] begin
    @testset for width_mult in WIDTH_MULTS, config in [:A1, :B1]
        m = MNASNet(config; width_mult) |> gpu
        @test size(m(x_224)) == (1000, 1)
        if (MNASNet, config, width_mult) in PRETRAINED_MODELS
            @test acctest(MNASNet(config; pretrain = true))
        else
            @test_throws ArgumentError MNASNet(config; pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "ConvNeXt" setup=[TestModels] begin
    configs = TEST_FAST ? [:small] : [:small, :base, :large, :tiny, :xlarge]
    @testset for config in configs
        m = ConvNeXt(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "ConvMixer" setup=[TestModels] begin
    configs = TEST_FAST ? [:small] : [:small, :base, :large]
    @testset for config in configs
        m = ConvMixer(config) |> gpu
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end

@testitem "UNet" setup=[TestModels] begin
    encoder = Metalhead.backbone(ResNet(18))
    model = UNet((256, 256), 3, 10, encoder) |> gpu
    @test size(model(x_256)) == (256, 256, 10, 1)
    @test gradtest(model, x_256)

    model = UNet() |> gpu
    @test size(model(x_256)) == (256, 256, 3, 1)
    _gc()
end
