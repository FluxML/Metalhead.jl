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
    ## TODO: find a way to port pretrained models to the new ResNet API
    # @testset "ResNet($sz)" for sz in [18, 34, 50, 101, 152]
        # if (ResNet, sz) in PRETRAINED_MODELS
        #     @test acctest(ResNet(sz, pretrain = true))
        # else
        #     @test_throws ArgumentError ResNet(sz, pretrain = true)
        # end
    # end

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
                    (dropout_rate = 0.1, drop_path_rate = 0.1, drop_block_rate = 0.1),
                    (dropout_rate = 0.5, drop_path_rate = 0.5, drop_block_rate = 0.5),
                    (dropout_rate = 0.8, drop_path_rate = 0.8, drop_block_rate = 0.8),
                ]
                @testset for drop_rates in drop_list
                    m = Metalhead.resnet(block_fn, layers; drop_rates...)
                    @test size(m(x_224)) == (1000, 1)
                    @test gradtest(m, x_224)
                    _gc()
                end
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
                    @test acctest(ResNeXt(depth, pretrain = true))
                else
                    @test_throws ArgumentError ResNeXt(depth, pretrain = true)
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

@testset "EfficientNet" begin
    @testset "EfficientNet($name)" for name in [:b0, :b1, :b2, :b3, :b4, :b5, :b6, :b7, :b8]
        # preferred image resolution scaling
        r = Metalhead.efficientnet_global_configs[name][1]
        x = rand(Float32, r, r, 3, 1)
        m = EfficientNet(name)
        @test size(m(x)) == (1000, 1)
        if (EfficientNet, name) in PRETRAINED_MODELS
            @test acctest(EfficientNet(name, pretrain = true))
        else
            @test_throws ArgumentError EfficientNet(name, pretrain = true)
        end
        @test gradtest(m, x)
        _gc()
    end
end

@testset "GoogLeNet" begin
    m = GoogLeNet()
    @test size(m(x_224)) == (1000, 1)
    if GoogLeNet in PRETRAINED_MODELS
        @test acctest(GoogLeNet(pretrain = true))
    else
        @test_throws ArgumentError GoogLeNet(pretrain = true)
    end
    @test gradtest(m, x_224)
    _gc()
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

@testset "MobileNet" verbose = true begin
    @testset "MobileNetv1" begin
        m = MobileNetv1()
        @test size(m(x_224)) == (1000, 1)
        if MobileNetv1 in PRETRAINED_MODELS
            @test acctest(MobileNetv1(pretrain = true))
        else
            @test_throws ArgumentError MobileNetv1(pretrain = true)
        end
        @test gradtest(m, x_224)
    end
    _gc()
    @testset "MobileNetv2" begin
        m = MobileNetv2()
        @test size(m(x_224)) == (1000, 1)
        if MobileNetv2 in PRETRAINED_MODELS
            @test acctest(MobileNetv2(pretrain = true))
        else
            @test_throws ArgumentError MobileNetv2(pretrain = true)
        end
        @test gradtest(m, x_224)
    end
    _gc()
    @testset "MobileNetv3" verbose = true begin
        @testset for mode in [:small, :large]
            m = MobileNetv3(mode)
            @test size(m(x_224)) == (1000, 1)
            if (MobileNetv3, mode) in PRETRAINED_MODELS
                @test acctest(MobileNetv3(mode; pretrain = true))
            else
                @test_throws ArgumentError MobileNetv3(mode; pretrain = true)
            end
            @test gradtest(m, x_224)
            _gc()
        end
    end
end


@testset "ConvNeXt" verbose = true begin
    @testset for mode in [:small, :base, :large, :tiny, :xlarge]
        @testset for drop_path_rate in [0.0, 0.5]
            m = ConvNeXt(mode; drop_path_rate)
            @test size(m(x_224)) == (1000, 1)
            @test gradtest(m, x_224)
            _gc()
        end
    end
end

@testset "ConvMixer" verbose = true begin
    @testset for mode in [:small, :base, :large]
        m = ConvMixer(mode)
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        _gc()
    end
end
