struct Bottleneck
  layer
end

@treelike Bottleneck

Bottleneck(in_planes, growth_rate) = Bottleneck(
                                          Chain(BatchNorm(in_planes, relu),
                                          Conv((1, 1), in_planes=>4growth_rate),
                                          BatchNorm(4growth_rate, relu),
                                          Conv((3, 3), 4growth_rate=>growth_rate, pad = (1, 1))))

(b::Bottleneck)(x) = cat(b.layer(x), x, dims = 3)

Transition(chs::Pair{<:Int, <:Int}) = Chain(BatchNorm(chs[1], relu),
                                            Conv((1, 1), chs),
                                            MeanPool((2, 2)))

function _make_dense_layers(block, in_planes, growth_rate, nblock)
  local layers = []
  for i in 1:nblock
    push!(layers, block(in_planes, growth_rate))
    in_planes += growth_rate
  end
  Chain(layers...)
end

function trained_densenet121_layers()
  weight = Metalhead.weights("densenet.bson")
  weights = Dict{Any, Any}()
  for ele in keys(weight)
    weights[string(ele)] = convert(Array{Float64, N} where N ,weight[ele])
  end
  ls = load_densenet(densenet_configs["densenet121"]...)
  ls[1].weight.data .= weights["conv1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
  ls[2].β.data .= weights["conv1/bn_b_0"]
  ls[2].γ.data .= weights["conv1/bn_w_0"]
  l = 4
  for (c, n) in enumerate([6, 12, 24, 16])
      for i in 1:n
          for j in [2, 4]
              ls[l][i].layer[j].weight.data .= weights["conv$(c+1)_$i/x$(j÷2)_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
              ls[l][i].layer[j-1].β.data .= weights["conv$(c+1)_$i/x$(j÷2)/bn_b_0"]
              ls[l][i].layer[j-1].γ.data .= weights["conv$(c+1)_$i/x$(j÷2)/bn_w_0"]
          end
      end
      l += 2
  end
  for i in [5, 7, 9] # Transition Block Conv Layers
    ls[i][2].weight.data .= weights["conv$(i÷2)_blk_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
    ls[i][1].β.data .= weights["conv$(i÷2)_blk/bn_b_0"]
    ls[i][1].γ.data .= weights["conv$(i÷2)_blk/bn_w_0"]
  end
  ls[end-1].W.data .= transpose(dropdims(weights["fc6_w_0"], dims = (1, 2))) # Dense Layers
  ls[end-1].b.data .= weights["fc6_b_0"]
  Flux.testmode!(ls)
  return ls
end

function load_densenet(block, nblocks; growth_rate = 32, reduction = 0.5, num_classes = 1000)
  num_planes = 2growth_rate
  layers = []
  push!(layers, Conv((7, 7), 3=>num_planes, stride = (2, 2), pad = (3, 3)))
  push!(layers, BatchNorm(num_planes, relu))
  push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))

  for i in 1:3
    push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[i]))
    num_planes += nblocks[i] * growth_rate
    out_planes = Int(floor(num_planes * reduction))
    push!(layers, Transition(num_planes=>out_planes))
    num_planes = out_planes
  end

  push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[4]))
  num_planes += nblocks[4] * growth_rate
  push!(layers, BatchNorm(num_planes, relu))

  Chain(layers..., MeanPool((7, 7)),
        x -> reshape(x, :, size(x, 4)),
        Dense(num_planes, num_classes), softmax)
end

densenet_configs =
  Dict("densenet121" => (Bottleneck, [6, 12, 24, 16]),
       "densenet169" => (Bottleneck, [6, 12, 32, 32]),
       "densenet201" => (Bottleneck, [6, 12, 48, 32]),
       "densenet264" => (Bottleneck, [6, 12, 64, 48]))

struct DenseNet121 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

DenseNet121() = DenseNet121(load_densenet(densenet_configs["densenet121"]...))

trained(::Type{DenseNet121}) = DenseNet121(trained_densenet121_layers())

Base.show(io::IO, ::DenseNet121) = print(io, "DenseNet264()")

@treelike DenseNet121

(m::DenseNet121)(x) = m.layers(x)

struct DenseNet169 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

DenseNet169() = DenseNet169(load_densenet(densenet_configs["densenet169"]...))

trained(::Type{DenseNet169}) = error("Pretrained Weights for DenseNet169 are not available")

Base.show(io::IO, ::DenseNet169) = print(io, "DenseNet169()")

@treelike DenseNet169

(m::DenseNet169)(x) = m.layers(x)

struct DenseNet201 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

DenseNet201() = DenseNet201(load_densenet(densenet_configs["densenet201"]...))

trained(::Type{DenseNet201}) = error("Pretrained Weights for DenseNet201 are not available")

Base.show(io::IO, ::DenseNet201) = print(io, "DenseNet201()")

@treelike DenseNet201

(m::DenseNet201)(x) = m.layers(x)

struct DenseNet264 <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

DenseNet264() = DenseNet264(load_densenet(densenet_configs["densenet264"]..., growth_rate=48))

trained(::Type{DenseNet264}) = error("Pretrained Weights for DenseNet264 are not available")

Base.show(io::IO, ::DenseNet264) = print(io, "DenseNet264()")

@treelike DenseNet264

(m::DenseNet264)(x) = m.layers(x)
