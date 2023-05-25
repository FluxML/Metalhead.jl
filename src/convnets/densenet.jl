function dense_bottleneck(inplanes::Integer, growth_rate::Integer, bn_size::Integer,
                          dropout_prob)
    return Chain(cat_channels,
                 conv_norm((1, 1), inplanes, bn_size * growth_rate;
                            revnorm = true)...,
                 conv_norm((3, 3), bn_size * growth_rate, growth_rate;
                            pad = 1, revnorm = true)...,
                 Dropout(dropout_prob))
end

function dense_block(inplanes::Integer, num_layers::Integer, bn_size::Integer,
                     growth_rate::Integer, dropout_prob)
    layers = [dense_bottleneck(inplanes + (i - 1) * growth_rate, growth_rate, bn_size,
                                   dropout_prob) for i in 1:num_layers]
    return DenseBlock(layers)
end

struct DenseBlock
    layers::Any
end
@functor DenseBlock

function (m::DenseBlock)(x)
    input = [x]
    for layer in m.layers
        x = layer(input)
        input = vcat(input, [x])
    end
    return cat_channels(input...)
end

"""
    transition(inplanes, outplanes)

Create a DenseNet transition sequence
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments

  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
"""
function transition(inplanes::Integer, outplanes::Integer)
    return Chain(conv_norm((1, 1), inplanes, outplanes; revnorm = true)...,
                 MeanPool((2, 2)))
end

function build_densenet(growth_rate::Integer, inplanes::Integer,
                        block_config::AbstractVector{<:Integer};
                        bn_size::Integer = 4, reduction = 0.5, dropout_prob = 0.0,
                        inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = []
    append!(layers,
            conv_norm((7, 7), inchannels, inplanes; stride = 2, pad = (3, 3)))
    push!(layers, MaxPool((3, 3); stride = 2, pad = (1, 1)))
    nfeatures = inplanes
    for (i, num_layers) in enumerate(block_config)
        push!(layers,
              dense_block(nfeatures, num_layers, bn_size, growth_rate, dropout_prob))
        nfeatures += num_layers * growth_rate
        if (i != length(block_config))
            push!(layers, transition(nfeatures, floor(Int, nfeatures * reduction)))
            nfeatures = floor(Int, nfeatures * reduction)
        end
    end
    push!(layers, BatchNorm(nfeatures, relu))
    return Chain(Chain(layers...), create_classifier(nfeatures, nclasses; dropout_prob))
end

function densenet(block_config::AbstractVector{<:Integer}; growth_rate::Integer = 32,
                  inplanes::Integer = 2 * growth_rate, dropout_prob = 0.0,
                  inchannels::Integer = 3, nclasses::Integer = 1000)
    return build_densenet(growth_rate, inplanes, block_config;
                          dropout_prob, inchannels, nclasses)
end

const DENSENET_CONFIGS = Dict(121 => [6, 12, 24, 16],
                              161 => [6, 12, 36, 24],
                              169 => [6, 12, 32, 32],
                              201 => [6, 12, 48, 32])

struct DenseNet
    layers::Any
end
@functor DenseNet

function DenseNet(config::Integer; pretrain::Bool = false, growth_rate::Integer = 32,
                  inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, keys(DENSENET_CONFIGS))
    layers = densenet(DENSENET_CONFIGS[config]; growth_rate, inchannels, nclasses)
    model = DenseNet(layers)
    if pretrain
        artifact_name = string("densenet", config)
        loadpretrain!(model, artifact_name) # see also HACK below
    end
    return model
end

(m::DenseNet)(x) = m.layers(x)

backbone(m::DenseNet) = m.layers[1]
classifier(m::DenseNet) = m.layers[2]

## HACK TO LOAD OLD WEIGHTS, remove when we have a new artifact
function Flux.loadmodel!(m::DenseNet, src)
    Flux.loadmodel!(m.layers[1], src.layers[1])
    return Flux.loadmodel!(m.layers[2], src.layers[2])
end
