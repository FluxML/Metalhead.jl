@testset "AlexNet" begin
    model = AlexNet()
    @test size(model(x_256)) == (1000, 1)
    @test_throws ArgumentError AlexNet(pretrain = true)
    @test gradtest(model, x_256)
    _gc()
end

@testset "VGG" begin
    @testset "VGG($sz, batchnorm=$bn)" for sz in [11, 13, 16, 19], bn in [true, false]
        m = VGG(sz, batchnorm = bn)
        @test size(m(x_224)) == (1000, 1)
        if (VGG, sz, bn) in PRETRAINED_MODELS
            @test acctest(VGG(sz, batchnorm = bn, pretrain = true))
        else
            @test_throws ArgumentError VGG(sz, batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "ResNet" begin
    # Tests for pretrained ResNets
    @testset "ResNet($sz)" for sz in [18, 34, 50, 101, 152]
        m = ResNet(sz)
        @test size(m(x_224)) == (1000, 1)
        if (ResNet, sz) in PRETRAINED_MODELS
            @test acctest(ResNet(sz, pretrain = true))
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
                [3, 8, 36, 3]
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

    @testset "WideResNet" begin
        @testset "WideResNet($sz)" for sz in [50, 101]
            m = WideResNet(sz)
            @test size(m(x_224)) == (1000, 1)
            @test gradtest(m, x_224)
            _gc()
            if (WideResNet, sz) in PRETRAINED_MODELS
                @test acctest(ResNet(sz, pretrain = true))
            else
                @test_throws ArgumentError WideResNet(sz, pretrain = true)
            end
        end
    end
end

@testset "ResNeXt" begin
    @testset for depth in [50, 101, 152]
        @testset for cardinality in [32, 64]
            @testset for base_width in [4, 8]
                m = ResNeXt(depth; cardinality, base_width)
                @test size(m(x_224)) == (1000, 1)
                if (ResNeXt, depth, cardinality, base_width) in PRETRAINED_MODELS
                    @test acctest(ResNeXt(depth; cardinality, base_width, pretrain = true))
                else
                    @test_throws ArgumentError ResNeXt(depth; cardinality, base_width, pretrain = true)
                end
                @test gradtest(m, x_224)
                _gc()
            end
        end
    end
end

@testset "SEResNet" begin
    @testset for depth in [18, 34, 50, 101, 152]
        m = SEResNet(depth)
        @test size(m(x_224)) == (1000, 1)
        if (SEResNet, depth) in PRETRAINED_MODELS
            @test acctest(SEResNet(depth, pretrain = true))
        else
            @test_throws ArgumentError SEResNet(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "SEResNeXt" begin
    @testset for depth in [50, 101, 152]
        @testset for cardinality in [32, 64]
            @testset for base_width in [4, 8]
                m = SEResNeXt(depth; cardinality, base_width)
                @test size(m(x_224)) == (1000, 1)
                if (SEResNeXt, depth, cardinality, base_width) in PRETRAINED_MODELS
                    @test acctest(SEResNeXt(depth, pretrain = true))
                else
                    @test_throws ArgumentError SEResNeXt(depth, pretrain = true)
                end
                @test gradtest(m, x_224)
                _gc()
            end
        end
    end
end

@testset "Res2Net" begin
    @testset for (base_width, scale) in [(26, 4), (48, 2), (14, 8), (26, 6), (26, 8)]
        m = Res2Net(50; base_width, scale)
        @test size(m(x_224)) == (1000, 1)
        if (Res2Net, depth, base_width, scale) in PRETRAINED_MODELS
            @test acctest(Res2Net(50; base_width, scale, pretrain = true))
        else
            @test_throws ArgumentError Res2Net(50; base_width, scale, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
    @testset for (base_width, scale) in [(26, 4)]
        m = Res2Net(101; base_width, scale)
        @test size(m(x_224)) == (1000, 1)
        if (Res2Net, depth, base_width, scale) in PRETRAINED_MODELS
            @test acctest(Res2Net(101; base_width, scale, pretrain = true))
        else
            @test_throws ArgumentError Res2Net(101; base_width, scale, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "Res2NeXt" begin
    @testset for depth in [50, 101]
        m = Res2NeXt(depth)
        @test size(m(x_224)) == (1000, 1)
        if (Res2NeXt, depth) in PRETRAINED_MODELS
            @test acctest(Res2NeXt(depth, pretrain = true))
        else
            @test_throws ArgumentError Res2NeXt(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "EfficientNet" begin
    @testset "EfficientNet($config)" for config in [:b0, :b1, :b2, :b3, :b4, :b5,] #:b6, :b7, :b8]
        # preferred image resolution scaling
        r = Metalhead.EFFICIENTNET_GLOBAL_CONFIGS[config][1]
        x = rand(Float32, r, r, 3, 1)
        m = EfficientNet(config)
        @test size(m(x)) == (1000, 1)
        if (EfficientNet, config) in PRETRAINED_MODELS
            @test acctest(EfficientNet(config, pretrain = true))
        else
            @test_throws ArgumentError EfficientNet(config, pretrain = true)
        end
        @test gradtest(m, x)
        _gc()
    end
end

@testset "EfficientNetv2" begin
    @testset for config in [:small, :medium, :large] # :xlarge]
        m = EfficientNetv2(config)
        @test size(m(x_224)) == (1000, 1)
        if (EfficientNetv2, config) in PRETRAINED_MODELS
            @test acctest(EfficientNetv2(config, pretrain = true))
        else
            @test_throws ArgumentError EfficientNetv2(config, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "GoogLeNet" begin
    @testset for bn in [true, false]
        m = GoogLeNet(batchnorm = bn)
        @test size(m(x_224)) == (1000, 1)
        if (GoogLeNet, bn) in PRETRAINED_MODELS
            @test acctest(GoogLeNet(batchnorm = bn, pretrain = true))
        else
            @test_throws ArgumentError GoogLeNet(batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "Inception" begin
    x_299 = rand(Float32, 299, 299, 3, 2)
    @testset "Inceptionv3" begin
        m = Inceptionv3()
        @test size(m(x_299)) == (1000, 2)
        if Inceptionv3 in PRETRAINED_MODELS
            @test acctest(Inceptionv3(pretrain = true))
        else
            @test_throws ArgumentError Inceptionv3(pretrain = true)
        end
        @test gradtest(m, x_299)
    end
    _gc()
    @testset "Inceptionv4" begin
        m = Inceptionv4()
        @test size(m(x_299)) == (1000, 2)
        if Inceptionv4 in PRETRAINED_MODELS
            @test acctest(Inceptionv4(pretrain = true))
        else
            @test_throws ArgumentError Inceptionv4(pretrain = true)
        end
        @test gradtest(m, x_299)
    end
    _gc()
    @testset "InceptionResNetv2" begin
        m = InceptionResNetv2()
        @test size(m(x_299)) == (1000, 2)
        if InceptionResNetv2 in PRETRAINED_MODELS
            @test acctest(InceptionResNetv2(pretrain = true))
        else
            @test_throws ArgumentError InceptionResNetv2(pretrain = true)
        end
        @test gradtest(m, x_299)
    end
    _gc()
    @testset "Xception" begin
        m = Xception()
        @test size(m(x_299)) == (1000, 2)
        if Xception in PRETRAINED_MODELS
            @test acctest(Xception(pretrain = true))
        else
            @test_throws ArgumentError Xception(pretrain = true)
        end
        @test gradtest(m, x_299)
    end
    _gc()
end

@testset "SqueezeNet" begin
    m = SqueezeNet()
    @test size(m(x_224)) == (1000, 1)
    if SqueezeNet in PRETRAINED_MODELS
        @test acctest(SqueezeNet(pretrain = true))
    else
        @test_throws ArgumentError SqueezeNet(pretrain = true)
    end
    @test gradtest(m, x_224)
    _gc()
end

@testset "DenseNet" begin
    @testset for sz in [121, 161, 169, 201]
        m = DenseNet(sz)
        @test size(m(x_224)) == (1000, 1)
        if (DenseNet, sz) in PRETRAINED_MODELS
            @test acctest(DenseNet(sz, pretrain = true))
        else
            @test_throws ArgumentError DenseNet(sz, pretrain = true)
        end
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "MobileNets (width = $width_mult)" for width_mult in [0.5, 0.75, 1, 1.3]
    @testset "MobileNetv1" begin
        m = MobileNetv1(width_mult)
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv1, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv1(pretrain = true))
        else
            @test_throws ArgumentError MobileNetv1(pretrain = true)
        end
        @test gradtest(m, x_224)
    end
    _gc()
    @testset "MobileNetv2" begin
        m = MobileNetv2(width_mult)
        @test size(m(x_224)) == (1000, 1)
        if (MobileNetv2, width_mult) in PRETRAINED_MODELS
            @test acctest(MobileNetv2(pretrain = true))
        else
            @test_throws ArgumentError MobileNetv2(pretrain = true)
        end
        @test gradtest(m, x_224)
    end
    _gc()
    @testset "MobileNetv3" verbose = true begin
        @testset for config in [:small, :large]
            m = MobileNetv3(config; width_mult)
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
    @testset "MNASNet" verbose = true begin
        @testset for config in [:A1, :B1]
            m = MNASNet(config; width_mult)
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
end

@testset "ConvNeXt" verbose = true begin
    @testset for config in [:small, :base, :large, :tiny, :xlarge]
        m = ConvNeXt(config)
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "ConvMixer" verbose = true begin
    @testset for config in [:small, :base, :large]
        m = ConvMixer(config)
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end

@testset "UNet" begin
    encoder = backbone(ResNet(18))
    model = UNet(encoder; inputsize = (256, 256, 3, 1), outplanes = 10)
    @test size(model(x_256)) == (256, 256, 10, 1)
    @test gradtest(model, x_256)
    _gc()
end
