# block configs for EfficientNet
# k: kernel size
# c: output channels
# e: expansion ratio
# s: stride
# n: number of repeats
# r: reduction ratio for squeeze-excite layer
# a: activation function
# Data is organised as (k, c, e, s, n, r, a)
const EFFICIENTNET_BLOCK_CONFIGS = [
    (mbconv, 3, 16, 1, 1, 1, 4, swish),
    (mbconv, 3, 24, 6, 2, 2, 4, swish),
    (mbconv, 5, 40, 6, 2, 2, 4, swish),
    (mbconv, 3, 80, 6, 2, 3, 4, swish),
    (mbconv, 5, 112, 6, 1, 3, 4, swish),
    (mbconv, 5, 192, 6, 2, 4, 4, swish),
    (mbconv, 3, 320, 6, 1, 1, 4, swish),
]

# Data is organised as (r, (w, d))
# r: image resolution
# w: width scaling
# d: depth scaling
const EFFICIENTNET_GLOBAL_CONFIGS = Dict(:b0 => (224, (1.0, 1.0)),
                                         :b1 => (240, (1.0, 1.1)),
                                         :b2 => (260, (1.1, 1.2)),
                                         :b3 => (300, (1.2, 1.4)),
                                         :b4 => (380, (1.4, 1.8)),
                                         :b5 => (456, (1.6, 2.2)),
                                         :b6 => (528, (1.8, 2.6)),
                                         :b7 => (600, (2.0, 3.1)),
                                         :b8 => (672, (2.2, 3.6)))

"""
    efficientnet(config::Symbol; norm_layer = BatchNorm, stochastic_depth_prob = 0.2,
                 dropout_prob = nothing, inchannels::Integer = 3, nclasses::Integer = 1000)

Create an EfficientNet model. ([reference](https://arxiv.org/abs/1905.11946v5)).

# Arguments

  - `config`: size of the model. Can be one of `[:b0, :b1, :b2, :b3, :b4, :b5, :b6, :b7, :b8]`.
  - `norm_layer`: normalization layer to use.
  - `stochastic_depth_prob`: probability of stochastic depth. Set to `nothing` to disable
    stochastic depth.
  - `dropout_prob`: probability of dropout in the classifier head. Set to `nothing` to disable
    dropout.
  - `inchannels`: number of input channels.
  - `nclasses`: number of output classes.
"""
function efficientnet(config::Symbol; norm_layer = BatchNorm, stochastic_depth_prob = 0.2,
                      dropout_prob = nothing, inchannels::Integer = 3,
                      nclasses::Integer = 1000)
    _checkconfig(config, keys(EFFICIENTNET_GLOBAL_CONFIGS))
    scalings = EFFICIENTNET_GLOBAL_CONFIGS[config][2]
    return build_invresmodel(scalings, EFFICIENTNET_BLOCK_CONFIGS; inplanes = 32,
                             norm_layer, stochastic_depth_prob, activation = swish,
                             headplanes = EFFICIENTNET_BLOCK_CONFIGS[end][3] * 4,
                             dropout_prob, inchannels, nclasses)
end

"""
    EfficientNet(config::Symbol; pretrain::Bool = false, inchannels::Integer = 3,
                 nclasses::Integer = 1000)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).

# Arguments

  - `config`: size of the model. Can be one of `[:b0, :b1, :b2, :b3, :b4, :b5, :b6, :b7, :b8]`.
  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels.
  - `nclasses`: number of output classes.

!!! warning

    EfficientNet does not currently support pretrained weights.

See also [`Metalhead.efficientnet`](@ref).
"""
struct EfficientNet
    layers::Any
end
@functor EfficientNet

function EfficientNet(config::Symbol; pretrain::Bool = false, inchannels::Integer = 3,
                      nclasses::Integer = 1000)
    layers = efficientnet(config; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("efficientnet-", config))
    end
    return EfficientNet(layers)
end

(m::EfficientNet)(x) = m.layers(x)

backbone(m::EfficientNet) = m.layers[1]
classifier(m::EfficientNet) = m.layers[2]
