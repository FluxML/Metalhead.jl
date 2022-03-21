using Metalhead, Test
using Flux

# PRETRAINED_MODELS = [(VGG19, false), ResNet50, GoogLeNet, DenseNet121, SqueezeNet]
PRETRAINED_MODELS = []

@testset "AlexNet" begin
  model = AlexNet()
  @test size(model(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
  @test_throws ArgumentError AlexNet(pretrain = true)
  @test_skip gradtest(model, rand(Float32, 256, 256, 3, 2))
end

@testset "VGG" begin
  @testset "$model(BN=$bn)" for model in [VGG11, VGG13, VGG16, VGG19], bn in [true, false]
    imsize = (224, 224)
    m = model(batchnorm = bn)

    @test size(m(rand(Float32, imsize..., 3, 2))) == (1000, 2)
    if (model, bn) in PRETRAINED_MODELS
      @test (model(batchnorm = bn, pretrain = true); true)
    else
      @test_throws ArgumentError model(batchnorm = bn, pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, imsize..., 3, 2))
  end
end

@testset "ResNet" begin
  @testset for model in [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]
    m = model()

    @test size(m(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
    if model in PRETRAINED_MODELS
      @test (model(pretrain = true); true)
    else
      @test_throws ArgumentError model(pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, 256, 256, 3, 2))
  end

  @testset "Shortcut C" begin
    m = Metalhead.resnet(Metalhead.basicblock, :C;
                         channel_config = [1, 1],
                         block_config = [2, 2, 2, 2])

    @test size(m(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
  end
end

@testset "ResNeXt" begin
  @testset for depth in [50, 101, 152]
    m = ResNeXt(depth)

    @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
    if ResNeXt in PRETRAINED_MODELS
      @test (ResNeXt(depth, pretrain = true); true)
    else
      @test_throws ArgumentError ResNeXt(depth, pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
  end
end

@testset "GoogLeNet" begin
  m = GoogLeNet()
  @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
  @test_throws ArgumentError (GoogLeNet(pretrain = true); true)
  @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
end

@testset "Inception3" begin
  m = Inception3()
  @test size(m(rand(Float32, 299, 299, 3, 2))) == (1000, 2)
  @test_throws ArgumentError Inception3(pretrain = true)
  @test_skip gradtest(m, rand(Float32, 299, 299, 3, 2))
end

@testset "SqueezeNet" begin
  m = SqueezeNet()
  @test size(m(rand(Float32, 227, 227, 3, 2))) == (1000, 2)
  @test_throws ArgumentError (SqueezeNet(pretrain = true); true)
  @test_skip gradtest(m, rand(Float32, 227, 227, 3, 2))
end

GC.gc()

@testset "DenseNet" begin
  @testset for model in [DenseNet121, DenseNet161, DenseNet169, DenseNet201]
    m = model()

    @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
    if model in PRETRAINED_MODELS
      @test (model(pretrain = true); true)
    else
      @test_throws ArgumentError model(pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
  end
end

@testset "MobileNet" verbose = true begin
  @testset "MobileNetv2" begin

    m = MobileNetv2()

    @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
    if MobileNetv2 in PRETRAINED_MODELS
      @test (MobileNetv2(pretrain = true); true)
    else
      @test_throws ArgumentError MobileNetv2(pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
  end

  @testset "MobileNetv3" verbose = true begin
    @testset for mode in [:small, :large]
      m = MobileNetv3(mode)

      @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
      if MobileNetv3 in PRETRAINED_MODELS
        @test (MobileNetv3(mode; pretrain = true); true)
      else
        @test_throws ArgumentError MobileNetv3(mode; pretrain = true)
      end
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
    end
  end
end

GC.gc()

@testset "ConvNeXt" verbose = true begin
  @testset for mode in [:tiny, :small, :base, :large, :xlarge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = ConvNeXt(mode; drop_path_rate)

      @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
    end
  end
end
