using Metalhead, Test
using Flux
using Flux: Zygote

PRETRAINED_MODELS = [(VGG19, false), ResNet50, GoogLeNet, DenseNet121, SqueezeNet]

function gradtest(model, input)
  y, pb = Zygote.pullback(() -> model(input), Flux.params(model))
  gs = pb(ones(Float32, size(y)))

  # if we make it to here with no error, success!
  return true
end

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
      @test_skip (model(batchnorm = bn, pretrain = true); true)
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
      @test_skip (model(pretrain = true); true)
    else
      @test_throws ArgumentError model(pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, 256, 256, 3, 2))
  end

  @testset "Shortcut C" begin
    m = Metalhead.resnet(block = Metalhead.basicblock,
                         channel_config = [1, 1],
                         block_config = [2, 2, 2, 2],
                         shortcut_config = :C)

    @test size(m(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
  end
end

@testset "GoogLeNet" begin
  m = GoogLeNet()
  @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
  @test_skip (GoogLeNet(pretrain = true); true)
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
  @test_skip (SqueezeNet(pretrain = true); true)
  @test_skip gradtest(m, rand(Float32, 227, 227, 3, 2))
end

@testset "DenseNet" begin
  @testset for model in [DenseNet121, DenseNet161, DenseNet169, DenseNet201]
    m = model()

    @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
    if model in PRETRAINED_MODELS
      @test_skip (model(pretrain = true); true)
    else
      @test_throws ArgumentError model(pretrain = true)
    end
    @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
  end
end
