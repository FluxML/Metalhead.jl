using Metalhead
using Pkg; Pkg.activate(@__DIR__)

include("pytorch2flux.jl")

const tvmodels = pyimport("torchvision.models")

# name, weight, jlconstructor, pyconstructor
model_list = [
              # ("vgg11", "IMAGENET1K_V1", () -> VGG(11), weights -> tvmodels.vgg11(; weights)),
              # ("vgg13", "IMAGENET1K_V1", () -> VGG(13), weights -> tvmodels.vgg13(; weights)),
              # ("vgg16", "IMAGENET1K_V1", () -> VGG(16), weights -> tvmodels.vgg16(; weights)),
              # ("vgg19", "IMAGENET1K_V1", () -> VGG(19), weights -> tvmodels.vgg19(; weights)),
              # ("resnet18", "IMAGENET1K_V1", () -> ResNet(18), weights -> tvmodels.resnet18(; weights)),
              # ("resnet34", "IMAGENET1K_V1", () -> ResNet(34), weights -> tvmodels.resnet34(; weights)),
              # ("resnet50", "IMAGENET1K_V1", () -> ResNet(50), weights -> tvmodels.resnet50(; weights)),
              # ("resnet101", "IMAGENET1K_V1", () -> ResNet(101), weights -> tvmodels.resnet101(; weights)),
              # ("resnet152", "IMAGENET1K_V1", () -> ResNet(152), weights -> tvmodels.resnet152(; weights)),
              # ("resnet50", "IMAGENET1K_V2", () -> ResNet(50), weights -> tvmodels.resnet50(; weights)),
              # ("resnet101", "IMAGENET1K_V2", () -> ResNet(101), weights -> tvmodels.resnet101(; weights)),
              # ("resnet152", "IMAGENET1K_V2", () -> ResNet(152), weights -> tvmodels.resnet152(; weights)),
              # ("resnext50_32x4d", "IMAGENET1K_V1", () -> ResNeXt(50; cardinality=32, base_width=4), weights -> tvmodels.resnext50_32x4d(; weights)),
              # ("resnext50_32x4d", "IMAGENET1K_V2", () -> ResNeXt(50; cardinality=32, base_width=4), weights -> tvmodels.resnext50_32x4d(; weights)),
              # ("resnext101_32x8d", "IMAGENET1K_V1", () -> ResNeXt(101; cardinality=32, base_width=8), weights -> tvmodels.resnext101_32x8d(; weights)),
              # ("resnext101_64x4d", "IMAGENET1K_V1", () -> ResNeXt(101; cardinality=64, base_width=4), weights -> tvmodels.resnext101_64x4d(; weights)),
              # ("resnext101_32x8d", "IMAGENET1K_V2", () -> ResNeXt(101; cardinality=32, base_width=8), weights -> tvmodels.resnext101_32x8d(; weights)),
              # ("wide_resnet50_2", "IMAGENET1K_V1", () -> WideResNet(50), weights -> tvmodels.wide_resnet50_2(; weights)),
              # ("wide_resnet50_2", "IMAGENET1K_V2", () -> WideResNet(50), weights -> tvmodels.wide_resnet50_2(; weights)),
              # ("wide_resnet101_2", "IMAGENET1K_V1", () -> WideResNet(101), weights -> tvmodels.wide_resnet101_2(; ; weights)),
              # ("wide_resnet101_2", "IMAGENET1K_V2", () -> WideResNet(101), weights -> tvmodels.wide_resnet101_2(; weights)),
              ("vit_b_16", "IMAGENET1K_V1", () -> ViT(:base, imsize=(224,224), qkv_bias=true), weights -> tvmodels.vit_b_16(; weights)),
              ## NOT MATCHING BELOW
              # ("squeezenet1_0", "IMAGENET1K_V1", () -> SqueezeNet(), weights -> tvmodels.squeezenet1_0(; weights)),
              # ("densenet121", "IMAGENET1K_V1", () -> DenseNet(121), weights -> tvmodels.densenet121(; weights)),
            ]


name, weights, jlconstructor, pyconstructor  = first(model_list)
# for (name, weights, jlconstructor, pyconstructor) in model_list
    jlmodel = jlconstructor()
    pymodel = pyconstructor(weights)
    pytorch2flux!(jlmodel, pymodel)
    compare_pytorch(jlmodel, pymodel)
    BSON.@save joinpath(@__DIR__, "$(name)_$weights.bson") model=jlmodel
    println("Saved $(name)_$weights.bson")
# end

using PythonCall

function vit_out_compare(jlmodel, pymodel)
  x = rand(Float32, 224, 224, 3, 1)

  conv16_class = jlmodel.layers[1][1:2]
  h = conv16_class(x)
  
  pymodel.eval()
  z = pymodel._process_input(jl2th(x))
  n = z.shape[0]
  batch_class_token = pymodel.class_token.expand(n, -1, -1)
  z = torch.cat(pylist([batch_class_token, z]), dim=1)
  
  @assert h ≈ np2jl(z.detach().numpy())
  
  encoder = jlmodel.layers[1][3:5]
  
  h = encoder[1](h)
  z = z + pymodel.encoder.pos_embedding
  @assert h ≈ np2jl(z.detach().numpy())
  
  h = encoder[2](h)
  z = pymodel.encoder.dropout(z)
  @assert h ≈ np2jl(z.detach().numpy())
 
  ###
  block = encoder[3][1]
  ln_1 = block[1].layers[1]
  hres = h
  h = ln_1(h)
  

  pyblock = pymodel.encoder.layers[0]
  zres = z
  z = pyblock.ln_1(z)
  @assert h ≈ np2jl(z.detach().numpy())
  ####
  mha = block[1].layers[2]
  h = mha(h)
  h = h + hres

  z, _ = pyblock.self_attention(z, z, z, need_weights=false)
  z = z + zres
  @assert h ≈ np2jl(z.detach().numpy())
  
  
  @assert h ≈ np2jl(z.detach().numpy())
end
  
vit_out_compare(jlmodel, pymodel)
  

