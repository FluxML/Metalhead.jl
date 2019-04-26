struct ResidualBlock
  layers
  shortcut
end

@treelike ResidualBlock

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
  layers = []
  for i in 2:length(filters)
    push!(layers, Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
    if i != length(filters)
      push!(layers, BatchNorm(filters[i], relu))
    else
      push!(layers, BatchNorm(filters[i]))
    end
  end
  ResidualBlock(Chain(layers...), shortcut)
end

ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity) =
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)

(r::ResidualBlock)(input) = relu.(r.layers(input) + r.shortcut(input))

function BasicBlock(filters::Int, downsample::Bool = false, res_top::Bool = false)
  # NOTE: res_top is set to true if this is the first residual connection of the architecture
  # If the number of channels is to be halved set the downsample argument to true
  if !downsample || res_top
    return ResidualBlock([filters for i in 1:3], [3,3], [1,1], [1,1])
  end
  shortcut = Chain(Conv((3,3), filters÷2=>filters, pad = (1,1), stride = (2,2)), BatchNorm(filters))
  ResidualBlock([filters÷2, filters, filters], [3,3], [1,1], [1,2], shortcut)
end

function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
  # NOTE: res_top is set to true if this is the first residual connection of the architecture
  # If the number of channels is to be halved set the downsample argument to true
  if !downsample && !res_top
    ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
  elseif downsample && res_top
    ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Chain(Conv((1,1), filters=>4 * filters, pad = (0,0), stride = (1,1)), BatchNorm(4 * filters)))
  else
    shortcut = Chain(Conv((1,1), 2 * filters=>4 * filters, pad = (0,0), stride = (2,2)), BatchNorm(4 * filters))
    ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
  end
end

function trained_resnet50_layers()
  weight = Metalhead.weights("resnet.bson")
  weights = Dict{Any ,Any}()
  for ele in keys(weight)
    weights[string(ele)] = weight[ele]
  end
  ls = load_resnet(resnet_configs["resnet50"]...)
  ls[1][1].weight.data .= flipkernel(weights["gpu_0/conv1_w_0"])
  ls[1][2].σ² .= weights["gpu_0/res_conv1_bn_riv_0"]
  ls[1][2].μ .= weights["gpu_0/res_conv1_bn_rm_0"]
  ls[1][2].β.data .= weights["gpu_0/res_conv1_bn_b_0"]
  ls[1][2].γ.data .= weights["gpu_0/res_conv1_bn_s_0"]
  count = 2
  for j in [3:5, 6:9, 10:15, 16:18]
    for p in j
      ls[p].layers[1].weight.data .= flipkernel(weights["gpu_0/res$(count)_$(p-j[1])_branch2a_w_0"])
      ls[p].layers[2].σ² .= weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_riv_0"]
      ls[p].layers[2].μ .= weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_rm_0"]
      ls[p].layers[2].β.data .= weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_b_0"]
      ls[p].layers[2].γ.data .= weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_s_0"]
      ls[p].layers[3].weight.data .= flipkernel(weights["gpu_0/res$(count)_$(p-j[1])_branch2b_w_0"])
      ls[p].layers[4].σ² .= weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_riv_0"]
      ls[p].layers[4].μ .= weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_rm_0"]
      ls[p].layers[4].β.data .= weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_b_0"]
      ls[p].layers[4].γ.data .= weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_s_0"]
      ls[p].layers[5].weight.data .= flipkernel(weights["gpu_0/res$(count)_$(p-j[1])_branch2c_w_0"])
      ls[p].layers[6].σ² .= weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_riv_0"]
      ls[p].layers[6].μ .= weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_rm_0"]
      ls[p].layers[6].β.data .= weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_b_0"]
      ls[p].layers[6].γ.data .= weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_s_0"]
    end
    ls[j[1]].shortcut[1].weight.data .= flipkernel(weights["gpu_0/res$(count)_0_branch1_w_0"])
    ls[j[1]].shortcut[2].σ² .= weights["gpu_0/res$(count)_0_branch1_bn_riv_0"]
    ls[j[1]].shortcut[2].μ .= weights["gpu_0/res$(count)_0_branch1_bn_rm_0"]
    ls[j[1]].shortcut[2].β.data .= weights["gpu_0/res$(count)_0_branch1_bn_b_0"]
    ls[j[1]].shortcut[2].γ.data .= weights["gpu_0/res$(count)_0_branch1_bn_s_0"]
    count += 1
  end
  ls[21].W.data .= transpose(weights["gpu_0/pred_w_0"]); ls[21].b.data .= weights["gpu_0/pred_b_0"]
  Flux.testmode!(ls)
  return ls
end

function load_resnet(Block, layers, initial_filters::Int = 64, nclasses::Int = 1000)
  local top = []
  local residual = []
  local bottom = []

  push!(top, Chain(Conv((7,7), 3=>initial_filters, pad = (3,3), stride = (2,2)),
                   BatchNorm(initial_filters)))
  push!(top, MaxPool((3,3), pad = (1,1), stride = (2,2)))

  for i in 1:length(layers)
    push!(residual, Block(initial_filters, true, i==1))
    for j in 2:layers[i]
      push!(residual, Block(initial_filters))
    end
    initial_filters *= 2
  end

  push!(bottom, MeanPool((7,7)))
  push!(bottom, x -> reshape(x, :, size(x,4)))
  if Block == Bottleneck
    push!(bottom, (Dense(2048, nclasses)))
  else
    push!(bottom, (Dense(512, nclasses)))
  end
  push!(bottom, softmax)

  Chain(top..., residual..., bottom...)
end

resnet_configs =
  Dict("resnet18" => (BasicBlock, [2, 2, 2, 2]),
       "resnet34" => (BasicBlock, [3, 4, 6, 3]),
       "resnet50" => (Bottleneck, [3, 4, 6, 3]),
       "resnet101" => (Bottleneck, [3, 4, 23, 3]),
       "resnet152" => (Bottleneck, [3, 8, 36, 3]))

struct ResNet18 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

ResNet18() = ResNet18(load_resnet(resnet_configs["resnet18"]...))

trained(::Type{ResNet18}) = error("Pretrained Weights for ResNet18 are not available")

Base.show(io::IO, ::ResNet18) = print(io, "ResNet18()")

@treelike ResNet18

(m::ResNet18)(x) = m.layers(x)

struct ResNet34 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

ResNet34() = ResNet34(load_resnet(resnet_configs["resnet34"]...))

trained(::Type{ResNet34}) = error("Pretrained Weights for ResNet34 are not available")

Base.show(io::IO, ::ResNet34) = print(io, "ResNet34()")

@treelike ResNet34

(m::ResNet34)(x) = m.layers(x)

struct ResNet50 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

ResNet50() = ResNet50(load_resnet(resnet_configs["resnet50"]...))

trained(::Type{ResNet50}) = ResNet50(trained_resnet50_layers())

Base.show(io::IO, ::ResNet50) = print(io, "ResNet50()")

@treelike ResNet50

(m::ResNet50)(x) = m.layers(x)

struct ResNet101 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

ResNet101() = ResNet101(load_resnet(resnet_configs["resnet101"]...))

trained(::Type{ResNet101}) = error("Pretrained Weights for ResNet101 are not available")

Base.show(io::IO, ::ResNet101) = print(io, "ResNet101()")

@treelike ResNet101

(m::ResNet101)(x) = m.layers(x)

struct ResNet152 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

ResNet152() = ResNet152(load_resnet(resnet_configs["resnet152"]...))

trained(::Type{ResNet152}) = error("Pretrained Weights for ResNet152 are not available")

Base.show(io::IO, ::ResNet152) = print(io, "ResNet152()")

@treelike ResNet152

(m::ResNet152)(x) = m.layers(x)
