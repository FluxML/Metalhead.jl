include("pytorch2flux.jl")

const tvmodels = pyimport("torchvision.models")

# name, weight, jlconstructor, pyconstructor
model_list = [
            #   ("vgg11", "IMAGENET1K_V1", () -> VGG(11), weights -> tvmodels.vgg11(weights=weights)),
            #   ("vgg13", "IMAGENET1K_V1", () -> VGG(13), weights -> tvmodels.vgg13(weights=weights)),
            #   ("vgg16", "IMAGENET1K_V1", () -> VGG(16), weights -> tvmodels.vgg16(weights=weights)),
            #   ("vgg19", "IMAGENET1K_V1", () -> VGG(19), weights -> tvmodels.vgg19(weights=weights)),
              ("resnet18", "IMAGENET1K_V1", () -> ResNet(18), weights -> tvmodels.resnet18(weights=weights)),
              ("resnet34", "IMAGENET1K_V1", () -> ResNet(34), weights -> tvmodels.resnet34(weights=weights)),
              ("resnet50", "IMAGENET1K_V1", () -> ResNet(50), weights -> tvmodels.resnet50(weights=weights)),
              ("resnet101", "IMAGENET1K_V1", () -> ResNet(101), weights -> tvmodels.resnet101(weights=weights)),
              ("resnet152", "IMAGENET1K_V1", () -> ResNet(152), weights -> tvmodels.resnet152(weights=weights)),
            #   ("resnet50", "IMAGENET1K_V2", () -> ResNet(50), weights -> tvmodels.resnet50(weights=weights)),
            #   ("resnet101", "IMAGENET1K_V2", () -> ResNet(101), weights -> tvmodels.resnet101(weights=weights)),
            #   ("resnet152", "IMAGENET1K_V2", () -> ResNet(152), weights -> tvmodels.resnet152(weights=weights)),
            #   ("resnext50_32x4d", "IMAGENET1K_V1", () -> ResNeXt(50, 32, 4), weights -> tvmodels.resnext50_32x4d(weights=weights)),
            #   ("resnext101_32x8d", "IMAGENET1K_V1", () -> ResNeXt(101, 32, 8), weights -> tvmodels.resnext101_32x8d(weights=weights)),
            #   ("wide_resnet50_2", "IMAGENET1K_V1", () -> WideResNet(50, 2), weights -> tvmodels.wide_resnet50_2(weights=weights)),
            #   ("wide_resnet101_2", "IMAGENET1K_V1", () -> WideResNet(101, 2), weights -> tvmodels.wide_resnet101_2(weights=weights)),
            #   ("densenet121", "IMAGENET1K_V1", () -> DenseNet(121), weights -> tvmodels.densenet121(weights=weights)),
            #   ("squeezenet", "IMAGENET1K_V1", () -> SqueezeNet(), weights -> tvmodels.squeezenet1_0(weights=weights)),
            ]


name, weights, jlconstructor, pyconstructor  = first(model_list)
# for (name, weights, jlconstructor, pyconstructor) in model_list
    jlmodel = jlconstructor()
    pymodel = pyconstructor(weights)
    pytorch2flux!(jlmodel, pymodel)
    compare_pytorch(jlmodel, pymodel)
    BSON.@save joinpath(@__DIR__,"$(name)_$weights.bson") model=jlmodel
    println("Saved $(name)_$weights.bson")
# end

