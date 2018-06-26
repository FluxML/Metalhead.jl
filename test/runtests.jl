using Metalhead
using Base.Test

vgg19 = VGG19()

testx = rand(Float32, 224, 224, 3, 1)

testy = vgg19(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

squeezenet = SqueezeNet()

testx = rand(Float32, 224, 224, 3, 1)

testy = squeezenet(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

densenet = DenseNet()

testx = rand(Float32, 224, 224, 3, 1)

testy = vgg19(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

resnet = ResNet()

testx = rand(Float32, 224, 224, 3, 1)

testy = vgg19(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

# Test proper download and functioning of CIFAR10
Metalhead.download(CIFAR10)
x1 = trainimgs(CIFAR10)[1]
x2 = valimgs(CIFAR10)[1]
@test size(x1.img) == (32, 32)
@test size(x2.img) == (32, 32)

# Just run the prediction code end-to-end
# TODO: Set up travis to actually run these
if length(datasets()) == 2
    for dataset in (ImageNet, CIFAR10)
        val1 = valimgs(dataset)[1]
        predict(vgg19, val1)
        classify(vgg19, val1)
    end
    predict(vgg19, testimgs(dataset(ImageNet))[1])
end
