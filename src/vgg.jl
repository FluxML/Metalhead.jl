function trained_vgg19_layers()
  ws = weights("vgg19.bson")
  ls = Chain(
    Conv(ws[:conv1_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv1_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv1_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv1_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    x -> maxpool(x, (2,2)),
    Conv(ws[:conv2_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv2_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv2_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv2_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    x -> maxpool(x, (2,2)),
    Conv(ws[:conv3_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv3_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv3_3_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv3_4_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    x -> maxpool(x, (2,2)),
    Conv(ws[:conv4_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv4_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv4_3_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv4_4_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    x -> maxpool(x, (2,2)),
    Conv(ws[:conv5_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv5_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv5_3_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv5_4_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    x -> maxpool(x, (2,2)),
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
      push!(layers, x -> maxpool(x, (2, 2)))
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

VGG11() = VGG11(load_vgg(vgg_configs["vgg11"]))

trained(::VGG11) = error("Pretrained Weights for VGG11 are not available")

Base.show(io::IO, ::VGG11) = print(io, "VGG11()")

Flux.treelike(VGG11)

(m::VGG11)(x) = m.layers(x)

struct VGG11_BN <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG11_BN() = VGG11_BN(load_vgg(vgg_configs["vgg11"], true))

trained(::VGG11_BN) = error("Pretrained Weights for VGG11_BN are not available")

Base.show(io::IO, ::VGG11_BN) = print(io, "VGG11_BN()")

Flux.treelike(VGG11_BN)

(m::VGG11_BN)(x) = m.layers(x)

struct VGG13 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG13() = VGG13(load_vgg(vgg_configs["vgg13"]))

trained(::VGG13) = error("Pretrained Weights for VGG13 are not available")

Base.show(io::IO, ::VGG13) = print(io, "VGG13()")

Flux.treelike(VGG13)

(m::VGG13)(x) = m.layers(x)

struct VGG13_BN <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG13_BN() = VGG13_BN(load_vgg(vgg_configs["vgg13"], true))

trained(::VGG13_BN) = error("Pretrained Weights for VGG13_BN are not available")

Base.show(io::IO, ::VGG13_BN) = print(io, "VGG13_BN()")

Flux.treelike(VGG13_BN)

(m::VGG13_BN)(x) = m.layers(x)

struct VGG16 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG16() = VGG16(load_vgg(vgg_configs["vgg16"]))

trained(::VGG16) = error("Pretrained Weights for VGG16 are not available")

Base.show(io::IO, ::VGG16) = print(io, "VGG16()")

Flux.treelike(VGG16)

(m::VGG16)(x) = m.layers(x)

struct VGG16_BN <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG16_BN() = VGG11_BN(load_vgg(vgg_configs["vgg16"], true))

trained(::VGG16_BN) = error("Pretrained Weights for VGG16_BN are not available")

Base.show(io::IO, ::VGG16_BN) = print(io, "VGG16_BN()")

Flux.treelike(VGG16_BN)

(m::VGG16_BN)(x) = m.layers(x)

struct VGG19 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG19() = VGG19(load_vgg(vgg_configs["vgg19"]))

trained(::VGG19) = VGG19(trained_vgg19_layers())

Base.show(io::IO, ::VGG19) = print(io, "VGG19()")

Flux.treelike(VGG19)

(m::VGG19)(x) = m.layers(x)

struct VGG19_BN <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG19_BN() = VGG11_BN(load_vgg(vgg_configs["vgg19"], true))

trained(::VGG19_BN) = error("Pretrained Weights for VGG19_BN are not available")

Base.show(io::IO, ::VGG19_BN) = print(io, "VGG19_BN()")

Flux.treelike(VGG19_BN)

(m::VGG19_BN)(x) = m.layers(x)
