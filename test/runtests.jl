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

testy = densenet(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

resnet = ResNet()

testx = rand(Float32, 224, 224, 3, 1)

testy = resnet(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

googlenet = GoogleNet()

testx = rand(Float32, 224, 224, 3, 1)

testy = googlenet(testx)

@test testy isa AbstractArray
@test length(testy) == 1000

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
