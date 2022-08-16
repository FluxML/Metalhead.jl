# block configs for EfficientNet
const EFFICIENTNET_BLOCK_CONFIGS = [
    # k, c, e, s, n, r, a
    (3, 16, 1, 1, 1, 4, swish),
    (3, 24, 6, 2, 2, 4, swish),
    (5, 40, 6, 2, 2, 4, swish),
    (3, 80, 6, 2, 3, 4, swish),
    (5, 112, 6, 1, 3, 4, swish),
    (5, 192, 6, 2, 4, 4, swish),
    (3, 320, 6, 1, 1, 4, swish),
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
    EfficientNet(config::Symbol; pretrain::Bool = false)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).
See also [`efficientnet`](#).

# Arguments

  - `config`: name of default configuration
    (can be `:b0`, `:b1`, `:b2`, `:b3`, `:b4`, `:b5`, `:b6`, `:b7`, `:b8`)
  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
"""
struct EfficientNet
    layers::Any
end
@functor EfficientNet

function EfficientNet(config::Symbol; pretrain::Bool = false, inchannels::Integer = 3,
                      nclasses::Integer = 1000)
    _checkconfig(config, keys(EFFICIENTNET_GLOBAL_CONFIGS))
    scalings = EFFICIENTNET_GLOBAL_CONFIGS[config][2]
    layers = efficientnet(EFFICIENTNET_BLOCK_CONFIGS,
                          fill(mbconv_builder, length(EFFICIENTNET_BLOCK_CONFIGS));
                          inplanes = 32, scalings, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("efficientnet-", config))
    end
    return EfficientNet(layers)
end

(m::EfficientNet)(x) = m.layers(x)

backbone(m::EfficientNet) = m.layers[1]
classifier(m::EfficientNet) = m.layers[2]
