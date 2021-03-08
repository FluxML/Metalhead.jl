using Metalhead, Test

PRETRAINED_MODELS = [vgg19, resnet50, googlenet, densenet121, squeezenet]

@testset "AlexNet" begin
  model = alexnet()
  @test size(model(rand(Float32, 256, 256, 3, 50))) == (1000, 50)
  @test_throws ArgumentError alexnet(pretrain=true)
end

@testset "VGG ($model)" for model in [vgg11, vgg11bn, vgg13, vgg13bn, vgg16, vgg16bn, vgg19, vgg19bn]
  imsize = (224, 224)
  m = model(imsize)  

  @test size(m(rand(Float32, imsize..., 3, 50))) == (1000, 50)
  if model in PRETRAINED_MODELS
    @test (model(imsize; pretrain=true); true)
  else
    @test_throws ArgumentError model(imsize; pretrain=true)
  end
end

@testset "ResNet ($model)" for model in [resnet18, resnet34, resnet50, resnet101, resnet152]
  m = model()
  @test size(m(rand(Float32, 256, 256, 3, 50))) == (1000, 50)

  if model in PRETRAINED_MODELS
    @test (model(pretrain=true); true)
  else
    @test_throws ArgumentError model(pretrain=true)
  end
end

@testset "GoogLeNet" begin
  m = googlenet()
  @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
  @test (googlenet(pretrain=true); true)
end

@testset "Inception3" begin
  m = inception3()
  @test size(m(rand(Float32, 299, 299, 3, 50))) == (1000, 50)
  @test_throws ArgumentError inception3(pretrain=true)
end

@testset "SqueezeNet" begin
  m = squeezenet()
  @test size(m(rand(Float32, 227, 227, 3, 50))) == (1000, 50)
  @test (squeezenet(pretrain=true); true)
end

@testset "DenseNet" for model in [densenet121, densenet161, densenet169, densenet201]
  m = model()
  @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
  
  if model in PRETRAINED_MODELS
    @test (model(pretrain=true); true)
  else
    @test_throws ArgumentError model(pretrain=true)
  end
end