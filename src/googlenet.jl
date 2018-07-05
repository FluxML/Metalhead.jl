struct InceptionBlock
  path_1
  path_2
  path_3
  path_4
end

Flux.treelike(InceptionBlock)

function InceptionBlock(in_chs, chs_1x1, chs_3x3_reduce, chs_3x3, chs_5x5_reduce, chs_5x5, pool_proj)
  path_1 = Conv((1, 1), in_chs=>chs_1x1, relu)

  path_2 = (Conv((1, 1), in_chs=>chs_3x3_reduce, relu),
            Conv((3, 3), chs_3x3_reduce=>chs_3x3, relu, pad = (1, 1)))

  path_3 = (Conv((1, 1), in_chs=>chs_5x5_reduce, relu),
            Conv((5, 5), chs_5x5_reduce=>chs_5x5, relu, pad = (2, 2)))

  path_4 = (x -> maxpool(x, (3,3), stride = (1, 1), pad = (1, 1)),
            Conv((1, 1), in_chs=>pool_proj, relu))

  InceptionBlock(path_1, path_2, path_3, path_4)
end

function (m::InceptionBlock)(x)
  cat(3, m.path_1(x), m.path_2[2](m.path_2[1](x)), m.path_3[2](m.path_3[1](x)), m.path_4[2](m.path_4[1](x)))
end

_googlenet() = Chain(Conv((7, 7), 3=>64, stride = (2, 2), relu, pad = (3, 3)),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      Conv((1, 1), 64=>64, relu),
      Conv((3, 3), 64=>192, relu, pad = (1, 1)),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      InceptionBlock(192, 64, 96, 128, 16, 32, 32),
      InceptionBlock(256, 128, 128, 192, 32, 96, 64),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      InceptionBlock(480, 192, 96, 208, 16, 48, 64),
      InceptionBlock(512, 160, 112, 224, 24, 64, 64),
      InceptionBlock(512, 128, 128, 256, 24, 64, 64),
      InceptionBlock(512, 112, 144, 288, 32, 64, 64),
      InceptionBlock(528, 256, 160, 320, 32, 128, 128),
      x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)),
      InceptionBlock(832, 256, 160, 320, 32, 128, 128),
      InceptionBlock(832, 384, 192, 384, 48, 128, 128),
      x -> meanpool(x, (7, 7), stride = (1, 1), pad = (0, 0)),
      x -> reshape(x, :, size(x, 4)),
      Dropout(0.4),
      Dense(1024, 1000), softmax)

function googlenet_layers()
  weight = Metalhead.weights("googlenet.bson")
  weights = Dict{Any, Any}()
  for ele in keys(weight)
    weights[string(ele)] = convert(Array{Float64, N} where N, weight[ele])
  end
  ls = _googlenet()
  ls[1].weight.data .= weights["conv1/7x7_s2_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[1].bias.data .= weights["conv1/7x7_s2_b_0"]
  ls[3].weight.data .= weights["conv2/3x3_reduce_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[3].bias.data .= weights["conv2/3x3_reduce_b_0"]
  ls[4].weight.data .= weights["conv2/3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[4].bias.data .= weights["conv2/3x3_b_0"]
  for (a, b) in [(6, "3a"), (7, "3b"), (9, "4a"), (10, "4b"), (11, "4c"), (12, "4d"), (13, "4e"), (15, "5a"), (16, "5b")]
    ls[a].path_1.weight.data .= weights["inception_$b/1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[a].path_1.bias.data .= weights["inception_$b/1x1_b_0"]
    ls[a].path_2[1].weight.data .= weights["inception_$b/3x3_reduce_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[a].path_2[1].bias.data .= weights["inception_$b/3x3_reduce_b_0"]
    ls[a].path_2[2].weight.data .= weights["inception_$b/3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[a].path_2[2].bias.data .= weights["inception_$b/3x3_b_0"]
    ls[a].path_3[1].weight.data .= weights["inception_$b/5x5_reduce_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[a].path_3[1].bias.data .= weights["inception_$b/5x5_reduce_b_0"]
    ls[a].path_3[2].weight.data .= weights["inception_$b/5x5_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[a].path_3[2].bias.data .= weights["inception_$b/5x5_b_0"]
    ls[a].path_4[2].weight.data .= weights["inception_$b/pool_proj_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]; ls[a].path_4[2].bias.data .= weights["inception_$b/pool_proj_b_0"]
  end
  ls[20].W.data .= transpose(weights["loss3/classifier_w_0"]); ls[20].b.data .= weights["loss3/classifier_b_0"]
  Flux.testmode!(ls)
  return ls
end

struct GoogleNet <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

GoogleNet() = GoogleNet(googlenet_layers())

Base.show(io::IO, ::GoogleNet) = print(io, "GoogleNet()")

Flux.treelike(GoogleNet)

(m::GoogleNet)(x) = m.layers(x)
