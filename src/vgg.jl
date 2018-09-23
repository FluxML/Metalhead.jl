function trained_vgg19_layers()
  ws = weights("vgg19.bson")
  ls = Chain(
    Conv(flipkernel(ws[:conv1_1_w_0]), ws[:conv1_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv1_2_w_0]), ws[:conv1_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(flipkernel(ws[:conv2_1_w_0]), ws[:conv2_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv2_2_w_0]), ws[:conv2_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(flipkernel(ws[:conv3_1_w_0]), ws[:conv3_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv3_2_w_0]), ws[:conv3_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv3_3_w_0]), ws[:conv3_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv3_4_w_0]), ws[:conv3_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(flipkernel(ws[:conv4_1_w_0]), ws[:conv4_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv4_2_w_0]), ws[:conv4_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv4_3_w_0]), ws[:conv4_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv4_4_w_0]), ws[:conv4_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(flipkernel(ws[:conv5_1_w_0]), ws[:conv5_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv5_2_w_0]), ws[:conv5_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv5_3_w_0]), ws[:conv5_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(flipkernel(ws[:conv5_4_w_0]), ws[:conv5_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(ws[:fc6_w_0]', ws[:fc6_b_0], relu),
    Dropout(0.5f0),
    Dense(ws[:fc7_w_0]', ws[:fc7_b_0], relu),
    Dropout(0.5f0),
    Dense(ws[:fc8_w_0]', ws[:fc8_b_0]),
    softmax)
  Flux.testmode!(ls)
  return ls
end

function load_vgg(arr, batchnorm::Bool = false)
  layers = []
  in_chs = 3
  for i in arr
    if i != 0
      push!(layers, Conv((3, 3), in_chs=>i, pad = (1, 1)))
      if batchnorm
        push!(layers, BatchNorm(i))
      end
      push!(layers, x -> relu.(x))
      in_chs = i
    else
      push!(layers, MaxPool(x, (2, 2)))
    end
  end
  push!(layers, [x -> reshape(x, :, size(x, 4)), Dense(25088, 4096, relu), Dropout(0.5),
                 Dense(4096, 4096, relu), Dropout(0.5), Dense(4096, 1000), softmax]...)
  Chain(layers...)
end

vgg_configs =
  Dict("vgg11" => [64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0],
       "vgg13" => [64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0],
       "vgg16" => [64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0],
       "vgg19" => [64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0])

struct VGG11 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG11(batchnorm::Bool = false) = VGG11(load_vgg(vgg_configs["vgg11"], batchnorm))

trained(::Type{VGG11}, batchnorm::Bool = false) =
  batchnorm ? error("Pretrained Weights for VGG11 BatchNorm are not available") : error("Pretrained Weights for VGG11 are not available")

Base.show(io::IO, ::VGG11) = print(io, "VGG11()")

@treelike VGG11

(m::VGG11)(x) = m.layers(x)

struct VGG13 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG13(batchnorm::Bool = false) = VGG13(load_vgg(vgg_configs["vgg13"], batchnorm))

trained(::Type{VGG13}, batchnorm::Bool = false) =
  batchnorm ? error("Pretrained Weights for VGG13 BatchNorm are not available") : error("Pretrained Weights for VGG13 are not available")

Base.show(io::IO, ::VGG13) = print(io, "VGG13()")

@treelike VGG13

(m::VGG13)(x) = m.layers(x)

struct VGG16 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG16(batchnorm::Bool = false) = VGG16(load_vgg(vgg_configs["vgg16"], batchnorm))

trained(::Type{VGG16}, batchnorm::Bool = false) =
  batchnorm ? error("Pretrained Weights for VGG16 BatchNorm are not available") : error("Pretrained Weights for VGG16 are not available")

Base.show(io::IO, ::VGG16) = print(io, "VGG16()")

@treelike VGG16

(m::VGG16)(x) = m.layers(x)

struct VGG19 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG19(batchnorm::Bool = false) = VGG19(load_vgg(vgg_configs["vgg19"], batchnorm))

trained(::Type{VGG19}, batchnorm::Bool = false) =
  batchnorm ? error("Pretrained Weights for VGG19 BatchNorm are not available") : VGG19(trained_vgg19_layers())

Base.show(io::IO, ::VGG19) = print(io, "VGG19()")

@treelike VGG19

(m::VGG19)(x) = m.layers(x)
