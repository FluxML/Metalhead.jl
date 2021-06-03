using Metalhead, Test

PRETRAINED_MODELS = [(VGG19, false), ResNet50, GoogLeNet, DenseNet121, SqueezeNet]

@testset "AlexNet" begin
  model = AlexNet()
  @test size(model(rand(Float32, 256, 256, 3, 50))) == (1000, 50)
  @test_throws ArgumentError AlexNet(pretrain=true)
end

@testset "VGG" begin
  @testset "$model{BN=$bn}" for model in [VGG11, VGG13, VGG16, VGG19], bn in [true, false]
    imsize = (224, 224)
    m = model(batchnorm=bn)

    @test size(m(rand(Float32, imsize..., 3, 50))) == (1000, 50)
    if (model, bn) in PRETRAINED_MODELS
      @test (model(batchnorm=bn, pretrain=true); true)
    else
      @test_throws ArgumentError model(batchnorm=bn, pretrain=true)
    end
  end
end

@testset "ResNet" begin
  @testset for model in [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]
    m = model()
    @test size(m(rand(Float32, 256, 256, 3, 50))) == (1000, 50)

    if model in PRETRAINED_MODELS
      @test (model(pretrain=true); true)
    else
      @test_throws ArgumentError model(pretrain=true)
    end
  end
end

@testset "GoogLeNet" begin
  m = GoogLeNet()
  @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
  @test (GoogLeNet(pretrain=true); true)
end

@testset "Inception3" begin
  m = Inception3()
  @test size(m(rand(Float32, 299, 299, 3, 50))) == (1000, 50)
  @test_throws ArgumentError Inception3(pretrain=true)
end

@testset "SqueezeNet" begin
  m = SqueezeNet()
  @test size(m(rand(Float32, 227, 227, 3, 50))) == (1000, 50)
  @test (SqueezeNet(pretrain=true); true)
end

@testset "DenseNet" for model in [DenseNet121, DenseNet161, DenseNet169, DenseNet201]
  m = model()
  @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
  
  if model in PRETRAINED_MODELS
    @test (model(pretrain=true); true)
  else
    @test_throws ArgumentError model(pretrain=true)
  end
end