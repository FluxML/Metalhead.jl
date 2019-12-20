function vgg19_layers()
  ws = weights("vgg19.bson")
  ls = Chain(
    Conv(ws[:conv1_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv1_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv1_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv1_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(ws[:conv2_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv2_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv2_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv2_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(ws[:conv3_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv3_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv3_3_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv3_4_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv3_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(ws[:conv4_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv4_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv4_3_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv4_4_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv4_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    Conv(ws[:conv5_1_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_1_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv5_2_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_2_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv5_3_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_3_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    Conv(ws[:conv5_4_w_0][end:-1:1,:,:,:][:,end:-1:1,:,:], ws[:conv5_4_b_0], relu, pad = (1,1), stride = (1,1), dilation = (1,1)),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(ws[:fc6_w_0]', ws[:fc6_b_0], relu),
    Dropout(0.5f0),
    Dense(ws[:fc7_w_0]', ws[:fc7_b_0], relu),
    Dropout(0.5f0),
    Dense(ws[:fc8_w_0]', ws[:fc8_b_0]),
    softmax)
  return ls
end

struct VGG19 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

VGG19() = VGG19(vgg19_layers())

Base.show(io::IO, ::VGG19) = print(io, "VGG19()")

@treelike VGG19

(m::VGG19)(x) = m.layers(x)
