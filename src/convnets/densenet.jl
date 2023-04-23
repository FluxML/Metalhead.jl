"""
    dense_bottleneck(inplanes, outplanes; expansion=4)

Create a Densenet bottleneck layer
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps on bottleneck branch
    (and scaling factor for inner feature maps; see ref)
"""
function dense_bottleneck(inplanes::Int, outplanes::Int; expansion::Int = 4)
    return SkipConnection(Chain(conv_norm((1, 1), inplanes, expansion * outplanes;
                                          revnorm = true)...,
                                conv_norm((3, 3), expansion * outplanes, outplanes;
                                          pad = 1, revnorm = true)...), cat_channels)
end

"""
    transition(inplanes, outplanes)

Create a DenseNet transition sequence
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
"""
function transition(inplanes::Int, outplanes::Int)
    return Chain(conv_norm((1, 1), inplanes, outplanes; revnorm = true)...,
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
function dense_block(inplanes::Int, growth_rates)
    return [dense_bottleneck(i, o)
            for (i, o) in zip(inplanes .+ cumsum([0, growth_rates[1:(end - 1)]...]),
                              growth_rates)]
end

"""
    densenet(inplanes, growth_rates; reduction = 0.5, dropout_prob = nothing, 
             inchannels = 3, nclasses = 1000)

Create a DenseNet model
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: the number of input feature maps to the first dense block
  - `growth_rates`: the growth rates of output feature maps within each
    [`dense_block`](@ref) (a vector of vectors)
  - `reduction`: the factor by which the number of feature maps is scaled across each transition
  - `dropout_prob`: the dropout probability for the classifier head. Set to `nothing` to disable dropout.
  - `nclasses`: the number of output classes
"""
function build_densenet(inplanes::Int, growth_rates; reduction = 0.5,
                        dropout_prob = nothing,
                        inchannels::Int = 3, nclasses::Int = 1000)
    layers = []
    append!(layers,
            conv_norm((7, 7), inchannels, inplanes; stride = 2, pad = (3, 3)))
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
    return Chain(Chain(layers...), create_classifier(outplanes, nclasses; dropout_prob))
end

"""
    densenet(nblocks::AbstractVector{Int}; growth_rate = 32,
             reduction = 0.5, dropout_prob = nothing, inchannels = 3,
             nclasses = 1000)

Create a DenseNet model
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `nblocks`: number of dense blocks between transitions
  - `growth_rate`: the output feature map growth probability of dense blocks (i.e. `k` in the ref)
  - `reduction`: the factor by which the number of feature maps is scaled across each transition
  - `dropout_prob`: the dropout probability for the classifier head. Set to `nothing` to disable dropout
  - `inchannels`: the number of input channels
  - `nclasses`: the number of output classes
"""
function densenet(nblocks::AbstractVector{Int}; growth_rate::Int = 32,
                  reduction = 0.5, dropout_prob = nothing, inchannels::Int = 3,
                  nclasses::Int = 1000)
    return build_densenet(2 * growth_rate, [fill(growth_rate, n) for n in nblocks];
                          reduction, dropout_prob, inchannels, nclasses)
end

const DENSENET_CONFIGS = Dict(121 => [6, 12, 24, 16],
                              161 => [6, 12, 36, 24],
                              169 => [6, 12, 32, 32],
                              201 => [6, 12, 48, 32])

"""
    DenseNet(config::Int; pretrain = false, growth_rate = 32,
             reduction = 0.5, inchannels = 3, nclasses = 1000)

Create a DenseNet model with specified configuration. Currently supported values are (121, 161, 169, 201)
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

    - `config`: the configuration of the model
    - `pretrain`: whether to load the model with pre-trained weights for ImageNet.
    - `growth_rate`: the output feature map growth probability of dense blocks (i.e. `k` in the ref)
    - `reduction`: the factor by which the number of feature maps is scaled across each transition
    - `inchannels`: the number of input channels
    - `nclasses`: the number of output classes

!!! warning
    
    `DenseNet` does not currently support pretrained weights.

See also [`Metalhead.densenet`](@ref).
"""
struct DenseNet
    layers::Any
end
@functor DenseNet

function DenseNet(config::Int; pretrain::Bool = false, growth_rate::Int = 32,
                  reduction = 0.5, inchannels::Int = 3, nclasses::Int = 1000)
    _checkconfig(config, keys(DENSENET_CONFIGS))
    layers = densenet(DENSENET_CONFIGS[config]; growth_rate, reduction, inchannels,
                      nclasses)
    if pretrain
        loadpretrain!(layers, string("densenet", config))
    end
    return DenseNet(layers)
end

(m::DenseNet)(x) = m.layers(x)

backbone(m::DenseNet) = m.layers[1]
classifier(m::DenseNet) = m.layers[2]
