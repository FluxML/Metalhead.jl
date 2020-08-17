struct ResidualBlock
  conv_layers
  norm_layers
  shortcut
end

@functor ResidualBlock

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
  local conv_layers = []
  local norm_layers = []
  for i in 2:length(filters)
    push!(conv_layers, Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
    push!(norm_layers, BatchNorm(filters[i]))
  end
  ResidualBlock(Tuple(conv_layers),Tuple(norm_layers),shortcut)
end

function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
  local value = copy.(input)
  for i in 1:length(block.conv_layers)-1
    value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
  end
  relu.(((block.norm_layers[end])((block.conv_layers[end])(value))) + block.shortcut(input))
end

function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
  if(!downsample && !res_top)
    return ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
  elseif(downsample && res_top)
    return ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Chain(Conv((1,1), filters=>4 * filters, pad = (0,0), stride = (1,1)), BatchNorm(4 * filters)))
  else
    shortcut = Chain(Conv((1,1), 2 * filters=>4 * filters, pad = (0,0), stride = (2,2)), BatchNorm(4 * filters))
    return ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
  end
end

function resnet(layers = [3,4,6,3])
  local layer_arr = []

  push!(layer_arr, Conv((7,7), 3=>64, pad = (3,3), stride = (2,2)))
  push!(layer_arr, MaxPool((3,3), pad = (1,1), stride = (2,2)))

  initial_filters = 64
  for i in 1:length(layers)
    push!(layer_arr, Bottleneck(initial_filters, true, i==1))
    for j in 2:layers[i]
      push!(layer_arr, Bottleneck(initial_filters))
    end
    initial_filters *= 2
  end

  push!(layer_arr, MeanPool((7,7)))
  push!(layer_arr, x -> reshape(x, :, size(x,4)))
  push!(layer_arr, (Dense(2048, 1000)))
  push!(layer_arr, softmax)

  Chain(layer_arr...)
end

function resnet_layers(layers = [3,4,6,3])
  # weight = Metalhead.weights("resnet.bson")
  # weights = Dict{Any ,Any}()
  # for ele in keys(weight)
  #   weights[string(ele)] = convert(Array{Float64, N} where N, weight[ele])
  # end
  ls = resnet(layers)
  # ls[1].weight .= weights["gpu_0/conv1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
  # count = 2
  # for j in [3:5, 6:9, 10:15, 16:18]
  #   for p in j
  #     ls[p].conv_layers[1].weight .= weights["gpu_0/res$(count)_$(p-j[1])_branch2a_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
  #     ls[p].conv_layers[2].weight .= weights["gpu_0/res$(count)_$(p-j[1])_branch2b_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
  #     ls[p].conv_layers[3].weight .= weights["gpu_0/res$(count)_$(p-j[1])_branch2c_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
  #   end
  #   count += 1
  # end
  # ls[21].W .= transpose(weights["gpu_0/pred_w_0"]); ls[21].b .= weights["gpu_0/pred_b_0"]
  return ls
end

struct ResNet <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

ResNet() = ResNet(resnet_layers())
ResNet50() = ResNet()
ResNet101() = ResNet(resnet_layers([3,4,23,3]))
ResNet152() = ResNet(resnet_layers([3,8,36,3]))

Base.show(io::IO, ::ResNet) = print(io, "ResNet()")

@functor ResNet

(m::ResNet)(x) = m.layers(x)
