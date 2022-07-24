"""
    dense_bottleneck(inplanes, growth_rate)

Create a Densenet bottleneck layer
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps on bottleneck branch
    (and scaling factor for inner feature maps; see ref)
"""
function dense_bottleneck(inplanes, outplanes)
    inner_channels = 4 * outplanes
    return SkipConnection(Chain(conv_norm((1, 1), inplanes, inner_channels; bias = false,
                                          prenorm = true)...,
                                conv_norm((3, 3), inner_channels, outplanes; pad = 1,
                                          bias = false, prenorm = true)...),
                          cat_channels)
end

"""
    transition(inplanes, outplanes)

Create a DenseNet transition sequence
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
"""
function transition(inplanes, outplanes)
    return Chain(conv_norm((1, 1), inplanes, outplanes; bias = false, prenorm = true)...,
                 MeanPool((2, 2)))
end

"""
    dense_block(inplanes, growth_rates)

Create a sequence of DenseNet bottlenecks increasing
the number of output feature maps by `growth_rates` with each block
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: number of input feature maps to the full sequence
  - `growth_rates`: the growth (additive) rates of output feature maps
    after each block (a vector of `k`s from the ref)
"""
function dense_block(inplanes, growth_rates)
    return [dense_bottleneck(i, o)
            for (i, o) in zip(inplanes .+ cumsum([0, growth_rates[1:(end - 1)]...]),
                              growth_rates)]
end

"""
    densenet(inplanes, growth_rates; reduction = 0.5, nclasses = 1000)

Create a DenseNet model
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: the number of input feature maps to the first dense block
  - `growth_rates`: the growth rates of output feature maps within each
    [`dense_block`](#) (a vector of vectors)
  - `reduction`: the factor by which the number of feature maps is scaled across each transition
  - `nclasses`: the number of output classes
"""
function densenet(inplanes, growth_rates; reduction = 0.5, nclasses = 1000)
    layers = []
    append!(layers, conv_norm((7, 7), 3, inplanes; stride = 2, pad = (3, 3), bias = false))
    push!(layers, MaxPool((3, 3); stride = 2, pad = (1, 1)))
    outplanes = 0
    for (i, rates) in enumerate(growth_rates)
        outplanes = inplanes + sum(rates)
        append!(layers, dense_block(inplanes, rates))
        (i != length(growth_rates)) &&
            push!(layers, transition(outplanes, floor(Int, outplanes * reduction)))
        inplanes = floor(Int, outplanes * reduction)
    end
    push!(layers, BatchNorm(outplanes, relu))
    return Chain(Chain(layers),
                 Chain(AdaptiveMeanPool((1, 1)),
                       MLUtils.flatten,
                       Dense(outplanes, nclasses)))
end

"""
    densenet(nblocks; growth_rate = 32, reduction = 0.5, nclasses = 1000)

Create a DenseNet model
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `nblocks`: number of dense blocks between transitions
  - `growth_rate`: the output feature map growth rate of dense blocks (i.e. `k` in the ref)
  - `reduction`: the factor by which the number of feature maps is scaled across each transition
  - `nclasses`: the number of output classes
"""
function densenet(nblocks::NTuple{N, <:Integer}; growth_rate = 32, reduction = 0.5,
                  nclasses = 1000) where {N}
    return densenet(2 * growth_rate, [fill(growth_rate, n) for n in nblocks];
                    reduction = reduction, nclasses = nclasses)
end

"""
    DenseNet(nblocks::NTuple{N, <:Integer};
             growth_rate = 32, reduction = 0.5, nclasses = 1000)

Create a DenseNet model
([reference](https://arxiv.org/abs/1608.06993)).
See also [`densenet`](#).

# Arguments

  - `nblocks`: number of dense blocks between transitions
  - `growth_rate`: the output feature map growth rate of dense blocks (i.e. `k` in the paper)
  - `reduction`: the factor by which the number of feature maps is scaled across each transition
  - `nclasses`: the number of output classes
"""
struct DenseNet
    layers::Any
end

function DenseNet(nblocks::NTuple{N, <:Integer};
                  growth_rate = 32, reduction = 0.5, nclasses = 1000) where {N}
    layers = densenet(nblocks; growth_rate = growth_rate,
                      reduction = reduction,
                      nclasses = nclasses)
    return DenseNet(layers)
end

@functor DenseNet

(m::DenseNet)(x) = m.layers(x)

backbone(m::DenseNet) = m.layers[1]
classifier(m::DenseNet) = m.layers[2]

const densenet_configs = Dict(121 => (6, 12, 24, 16),
                              161 => (6, 12, 36, 24),
                              169 => (6, 12, 32, 32),
                              201 => (6, 12, 48, 32))

"""
    DenseNet(config::Integer = 121; pretrain = false, nclasses = 1000)
    DenseNet(transition_configs::NTuple{N,Integer})

Create a DenseNet model with specified configuration. Currently supported values are (121, 161, 169, 201)
([reference](https://arxiv.org/abs/1608.06993)).
Set `pretrain = true` to load the model with pre-trained weights for ImageNet.

!!! warning
    
    `DenseNet` does not currently support pretrained weights.

See also [`Metalhead.densenet`](#).
"""
function DenseNet(config::Integer = 121; pretrain = false, nclasses = 1000)
    _checkconfig(config, keys(densenet_configs))
    model = DenseNet(densenet_configs[config]; nclasses = nclasses)
    if pretrain
        loadpretrain!(model, string("DenseNet", config))
    end
    return model
end
