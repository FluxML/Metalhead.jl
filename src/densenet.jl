struct Bottleneck
    layer
end

@functor Bottleneck

Bottleneck(in_planes, growth_rate) = Bottleneck(Chain(BatchNorm(in_planes, relu),
                                          Conv((1, 1), in_planes => 4growth_rate),
                                          BatchNorm(4growth_rate, relu),
                                          Conv((3, 3), 4growth_rate => growth_rate, pad = (1, 1))))

(b::Bottleneck)(x) = cat(b.layer(x), x, dims = 3)

Transition(chs::Pair{<:Int,<:Int}) = Chain(BatchNorm(chs[1], relu),
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

function _densenet(nblocks = [6, 12, 24, 16]; block = Bottleneck, growth_rate = 32, reduction = 0.5, num_classes = 1000)
    num_planes = 2growth_rate
    layers = []
    push!(layers, Conv((7, 7), 3 => num_planes, stride = (2, 2), pad = (3, 3)))
    push!(layers, BatchNorm(num_planes, relu))
    push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))

    for i in 1:3
        push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[i]))
        num_planes += nblocks[i] * growth_rate
        out_planes = Int(floor(num_planes * reduction))
        push!(layers, Transition(num_planes => out_planes))
        num_planes = out_planes
    end

    push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[4]))
    num_planes += nblocks[4] * growth_rate
    push!(layers, BatchNorm(num_planes, relu))

    Chain(layers..., MeanPool((7, 7)),
        x->reshape(x, :, size(x, 4)),
        Dense(num_planes, num_classes), softmax)
end

function densenet_layers()
    weight = Metalhead.weights("densenet.bson")
    weights = Dict{Any,Any}()
    for ele in keys(weight)
        weights[string(ele)] = convert(Array{Float64,N} where N, weight[ele])
    end
    ls = _densenet()
    ls[1].weight .= weights["conv1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
    ls[2].β .= weights["conv1/bn_b_0"]
    ls[2].γ .= weights["conv1/bn_w_0"]
    l = 4
    for (c, n) in enumerate([6, 12, 24, 16])
        for i in 1:n
            for j in [2, 4]
                ls[l][i].layer[j].weight .= weights["conv$(c + 1)_$i/x$(j ÷ 2)_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
                ls[l][i].layer[j - 1].β .= weights["conv$(c + 1)_$i/x$(j ÷ 2)/bn_b_0"]
                ls[l][i].layer[j - 1].γ .= weights["conv$(c + 1)_$i/x$(j ÷ 2)/bn_w_0"]
            end
        end
        l += 2
    end
    for i in [5, 7, 9] # Transition Block Conv Layers
        ls[i][2].weight .= weights["conv$(i ÷ 2)_blk_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
        ls[i][1].β .= weights["conv$(i ÷ 2)_blk/bn_b_0"]
        ls[i][1].γ .= weights["conv$(i ÷ 2)_blk/bn_w_0"]
    end
    ls[end - 1].W .= transpose(dropdims(weights["fc6_w_0"], dims = (1, 2))) # Dense Layers
    ls[end - 1].b .= weights["fc6_b_0"]
    return ls
end

struct DenseNet <: ClassificationModel{ImageNet.ImageNet1k}
    layers::Chain
end

DenseNet() = DenseNet(densenet_layers())

Base.show(io::IO, ::DenseNet) = print(io, "DenseNet()")

@functor DenseNet

(m::DenseNet)(x) = m.layers(x)
