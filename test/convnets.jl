using Metalhead, Test
using Flux

# PRETRAINED_MODELS = [(VGG19, false), ResNet50, GoogLeNet, DenseNet121, SqueezeNet]
PRETRAINED_MODELS = []

@testset "AlexNet" begin
    model = AlexNet()
    @test size(model(x_256)) == (1000, 1)
    @test_throws ArgumentError AlexNet(pretrain = true)
    @test gradtest(model, x_256)
end

GC.safepoint()
GC.gc()

@testset "VGG" begin 
    @testset "VGG($sz, batchnorm=$bn)" for sz in [11, 13, 16, 19], bn in [true, false]
        m = VGG(sz, batchnorm = bn)
        @test size(m(x_224)) == (1000, 1)
        if (VGG, sz, bn) in PRETRAINED_MODELS
            @test (VGG(sz, batchnorm = bn, pretrain = true); true)
        else
            @test_throws ArgumentError VGG(sz, batchnorm = bn, pretrain = true)
        end
        @test gradtest(m, x_224)
        GC.safepoint()
        GC.gc()
    end
end

GC.safepoint()
GC.gc()

@testset "ResNet" begin
    @testset "ResNet($sz)" for sz in [18, 34, 50, 101, 152]
        m = ResNet(sz)
        @test size(m(x_256)) == (1000, 1)
        if (ResNet, sz) in PRETRAINED_MODELS
            @test (ResNet(sz, pretrain = true); true)
        else
            @test_throws ArgumentError ResNet(sz, pretrain = true)
        end
        @test gradtest(m, x_256)
        GC.safepoint()
        GC.gc()
    end

    @testset "Shortcut C" begin
        m = Metalhead.resnet(Metalhead.basicblock, :C;
                             channel_config = [1, 1],
                             block_config = [2, 2, 2, 2])
        @test size(m(x_256)) == (1000, 1)
        @test gradtest(m, x_256)
    end
end

GC.safepoint()
GC.gc()

@testset "ResNeXt" begin 
    @testset for depth in [50, 101, 152]
        m = ResNeXt(depth)
        @test size(m(x_224)) == (1000, 1)
        if ResNeXt in PRETRAINED_MODELS
            @test (ResNeXt(depth, pretrain = true); true)
        else
            @test_throws ArgumentError ResNeXt(depth, pretrain = true)
        end
        @test gradtest(m, x_224)
        GC.safepoint()
        GC.gc()
    end
end

GC.safepoint()
GC.gc()

@testset "GoogLeNet" begin
    m = GoogLeNet()
    @test size(m(x_224)) == (1000, 1)
    @test_throws ArgumentError (GoogLeNet(pretrain = true); true)
    @test gradtest(m, x_224)
end

GC.safepoint()
GC.gc()

@testset "Inception3" begin
    m = Inception3()
    @test size(m(x_224)) == (1000, 1)
    @test_throws ArgumentError Inception3(pretrain = true)
    @test gradtest(m, x_224)
end

GC.safepoint()
GC.gc()

@testset "SqueezeNet" begin
    m = SqueezeNet()
    @test size(m(x_224)) == (1000, 1)
    @test_throws ArgumentError (SqueezeNet(pretrain = true); true)
    @test gradtest(m, x_224)
end

GC.safepoint()
GC.gc()

@testset "DenseNet" begin 
    @testset for sz in [121, 161, 169, 201]
        m = DenseNet(sz)
        @test size(m(x_224)) == (1000, 1)
        if (DenseNet, sz) in PRETRAINED_MODELS
            @test (DenseNet(sz, pretrain = true); true)
        else
            @test_throws ArgumentError DenseNet(sz, pretrain = true)
        end
        @test gradtest(m, x_224)
        GC.safepoint()
        GC.gc()
    end
end

GC.safepoint()
GC.gc()

@testset "MobileNet" verbose=true begin
    @testset "MobileNetv1" begin
        m = MobileNetv1()
        @test size(m(x_224)) == (1000, 1)
        if MobileNetv1 in PRETRAINED_MODELS
            @test (MobileNetv1(pretrain = true); true)
        else
            @test_throws ArgumentError MobileNetv1(pretrain = true)
        end
        @test gradtest(m, x_224)
    end

    GC.safepoint()
    GC.gc()

    @testset "MobileNetv2" begin
        m = MobileNetv2()
        @test size(m(x_224)) == (1000, 1)
        if MobileNetv2 in PRETRAINED_MODELS
            @test (MobileNetv2(pretrain = true); true)
        else
            @test_throws ArgumentError MobileNetv2(pretrain = true)
        end
        @test gradtest(m, x_224)
    end

    GC.safepoint()
    GC.gc()

    @testset "MobileNetv3" verbose=true begin
        @testset for mode in [:small, :large]
            m = MobileNetv3(mode)
            @test size(m(x_224)) == (1000, 1)
            if MobileNetv3 in PRETRAINED_MODELS
                @test (MobileNetv3(mode; pretrain = true); true)
            else
                @test_throws ArgumentError MobileNetv3(mode; pretrain = true)
            end
            @test gradtest(m, x_224)
        end
    end
end

GC.safepoint()
GC.gc()

@testset "ConvNeXt" verbose=true begin
    @testset for mode in [:small, :base, :large] # :tiny, #, :xlarge]
        @testset for drop_path_rate in [0.0, 0.5]
            m = ConvNeXt(mode; drop_path_rate)
            @test size(m(x_224)) == (1000, 1)
            @test gradtest(m, x_224)
            GC.safepoint()
            GC.gc()
        end
    end
end

GC.safepoint()
GC.gc()

@testset "ConvMixer" verbose=true begin
    @testset for mode in [:small, :base, :large]
        m = ConvMixer(mode)
        @test size(m(x_224)) == (1000, 1)
        @test gradtest(m, x_224)
        GC.safepoint()
        GC.gc()
    end
end
