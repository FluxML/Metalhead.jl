using Metalhead
using Pkg; Pkg.activate(@__DIR__)
using JLD2

include("pytorch2flux.jl")

const tvmodels = pyimport("torchvision.models")

# name, weight, jlconstructor, pyconstructor
model_list = [
              ("vgg11", "IMAGENET1K_V1", () -> VGG(11, batchnorm=false), weights -> tvmodels.vgg11(; weights)),
              ("vgg13", "IMAGENET1K_V1", () -> VGG(13, batchnorm=false), weights -> tvmodels.vgg13(; weights)),
              ("vgg16", "IMAGENET1K_V1", () -> VGG(16, batchnorm=false), weights -> tvmodels.vgg16(; weights)),
              ("vgg19", "IMAGENET1K_V1", () -> VGG(19, batchnorm=false), weights -> tvmodels.vgg19(; weights)),
              ("resnet18", "IMAGENET1K_V1", () -> ResNet(18), weights -> tvmodels.resnet18(; weights)),
              ("resnet34", "IMAGENET1K_V1", () -> ResNet(34), weights -> tvmodels.resnet34(; weights)),
              ("resnet50", "IMAGENET1K_V1", () -> ResNet(50), weights -> tvmodels.resnet50(; weights)),
              ("resnet50", "IMAGENET1K_V2", () -> ResNet(50), weights -> tvmodels.resnet50(; weights)),
              ("resnet101", "IMAGENET1K_V1", () -> ResNet(101), weights -> tvmodels.resnet101(; weights)),
              ("resnet101", "IMAGENET1K_V2", () -> ResNet(101), weights -> tvmodels.resnet101(; weights)),
              ("resnext50_32x4d", "IMAGENET1K_V1", () -> ResNeXt(50; cardinality=32, base_width=4), weights -> tvmodels.resnext50_32x4d(; weights)),
              ("resnext50_32x4d", "IMAGENET1K_V2", () -> ResNeXt(50; cardinality=32, base_width=4), weights -> tvmodels.resnext50_32x4d(; weights)),
              ("resnext101_32x8d", "IMAGENET1K_V1", () -> ResNeXt(101; cardinality=32, base_width=8), weights -> tvmodels.resnext101_32x8d(; weights)),
              ("resnext101_32x8d", "IMAGENET1K_V2", () -> ResNeXt(101; cardinality=32, base_width=8), weights -> tvmodels.resnext101_32x8d(; weights)),
              ("resnext101_64x4d", "IMAGENET1K_V1", () -> ResNeXt(101; cardinality=64, base_width=4), weights -> tvmodels.resnext101_64x4d(; weights)),
              ("resnet152", "IMAGENET1K_V1", () -> ResNet(152), weights -> tvmodels.resnet152(; weights)),
              ("resnet152", "IMAGENET1K_V2", () -> ResNet(152), weights -> tvmodels.resnet152(; weights)),
              ("wideresnet50", "IMAGENET1K_V1", () -> WideResNet(50), weights -> tvmodels.wide_resnet50_2(; weights)),
              ("wideresnet50", "IMAGENET1K_V2", () -> WideResNet(50), weights -> tvmodels.wide_resnet50_2(; weights)),
              ("wideresnet101", "IMAGENET1K_V1", () -> WideResNet(101), weights -> tvmodels.wide_resnet101_2(; weights)),
              ("wideresnet101", "IMAGENET1K_V2", () -> WideResNet(101), weights -> tvmodels.wide_resnet101_2(; weights)),
              ("vit_b_16", "IMAGENET1K_V1", () -> ViT(:base), weights -> tvmodels.vit_b_16(; weights)),
              ("vit_b_32", "IMAGENET1K_V1", () -> ViT(:base, patch_size=(32,32)), weights -> tvmodels.vit_b_32(; weights)),
              ("vit_l_16", "IMAGENET1K_V1", () -> ViT(:large), weights -> tvmodels.vit_l_16(; weights)),
              ("vit_l_32", "IMAGENET1K_V1", () -> ViT(:large, patch_size=(32,32)), weights -> tvmodels.vit_l_32(; weights)),
              ## NOT WORKING:
              # ("densenet121", "IMAGENET1K_V1", () -> DenseNet(121), weights -> tvmodels.densenet121(; weights)),
              # ("squeezenet1_0", "IMAGENET1K_V1", () -> SqueezeNet(), weights -> tvmodels.squeezenet1_0(; weights)),
              # ("vit_h_14", "IMAGENET1K_SWAG_E2E_V1", () -> ViT(:huge, imsize=(224,224), patch_size=(14,14), qkv_bias=true), weights -> tvmodels.vit_h_14(; weights)),
              # ("vit_h_14", "IMAGENET1K_SWAG_LINEAR_V1", () -> ViT(:huge, imsize=(224,224), patch_size=(14,14), qkv_bias=true), weights -> tvmodels.vit_h_14(; weights)),
              # ("vgg11_bn", "IMAGENET1K_V1", () -> VGG(11, batchnorm=true), weights -> tvmodels.vgg11_bn(; weights)),
              # ("vgg13_bn", "IMAGENET1K_V1", () -> VGG(13, batchnorm=true), weights -> tvmodels.vgg13_bn(; weights)),
              # ("vgg16_bn", "IMAGENET1K_V1", () -> VGG(16, batchnorm=true), weights -> tvmodels.vgg16_bn(; weights)),
              # ("vgg19_bn", "IMAGENET1K_V1", () -> VGG(19, batchnorm=true), weights -> tvmodels.vgg19_bn(; weights)),
            ]


function save_model_state(filename, model)
    mkpath(dirname(filename))
    if endswith(filename, ".jld2")
        JLD2.jldsave(filename, model_state = Flux.state(model))
    elseif endswith(filename, ".bson")
        BSON.@save filename model_state=Flux.state(model)
    else
        error("Unknown file extension")
    end
end

function load_model_state(filename)
    if endswith(filename, ".jld2")
        return JLD2.load(filename)["model_state"]
    elseif endswith(filename, ".bson")
        return BSON.load(filename)[:model_state]
    else
        error("Unknown file extension")
    end
end

function convert_models()
  # name, weights, jlconstructor, pyconstructor  = first(model_list)
    for (name, weights, jlconstructor, pyconstructor) in model_list
        # CONSTRUCT MODELS
        jlmodel = jlconstructor()
        pymodel = pyconstructor(weights)

        # LOAD WEIGHTS FROM PYTORCH TO JULIA
        pytorch2flux!(jlmodel, pymodel)
        rtol = startswith(name, "vit") ? 1e-2 : 1e-4 # TODO investigate why ViT is less accurate
        compare_pytorch(jlmodel, pymodel; rtol)
        
        # SAVE WEIGHTS
        artifact_name = "$(name)-$weights"
        filename = joinpath(@__DIR__, "weights", name, artifact_name, "$(artifact_name)")
        if name != "resnet152"
            filename *= ".jld2"
        else
            filename *= ".bson" # TODO: fix resnet152.jld2, not sure why it's not working
        end
        save_model_state(filename, jlmodel)
        println("Saved $filename")

        # LOAD WEIGHTS AND TEST AGAIN
        jlmodel2 = jlconstructor()
        model_state = load_model_state(filename)
        Flux.loadmodel!(jlmodel2, model_state)
        compare_pytorch(jlmodel2, pymodel; rtol)
    end
end

convert_models()
