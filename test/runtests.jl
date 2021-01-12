using Metalhead, Test

@testset "AlexNet" begin
    model = alexnet()
    @test size(model(rand(Float32, 256, 256, 3, 50))) == (1000, 50)
end

@testset "VGG ($model)" for model in [vgg11, vgg11bn, vgg13, vgg13bn, vgg16, vgg16bn, vgg19, vgg19bn]
    imsize = (224, 224)
    m = model(imsize)  
  
    @test size(m(rand(Float32, imsize..., 3, 50))) == (1000, 50)
end

@testset "ResNet ($model)" for model in [resnet18, resnet34, resnet50, resnet101, resnet152]
    m = model()
    @test size(m(rand(Float32, 256, 256, 3, 50))) == (1000, 50)
end

@testset "GoogLeNet" begin
    m = googlenet()
    @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
end

@testset "Inception3" begin
    m = inception3()
    @test size(m(rand(Float32, 299, 299, 3, 50))) == (1000, 50)
end

@testset "SqueezeNet" begin
    m = squeezenet()
    @test size(m(rand(Float32, 227, 227, 3, 50))) == (1000, 50)
end

@testset "DenseNet" begin
    for model in [densenet_121, densenet_161, densenet_169, densenet_201]
        m = model()
        @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
    end
end