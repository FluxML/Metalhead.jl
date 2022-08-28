# Layer configurations for MNasNet
# f: block function - we use `dwsep_conv_norm` for the first block and `mbconv` for the rest
# k: kernel size
# c: output channels
# e: expansion factor - only used for `mbconv`
# s: stride
# n: number of repeats
# r: reduction factor - only used for `mbconv`
# a: activation function
# Data is organised as (f, k, c, (e,) s, n, (r,) a)
const MNASNET_CONFIGS = Dict(:B1 => (32,
                                     [
                                         (dwsep_conv_norm, 3, 16, 1, 1, relu),
                                         (mbconv, 3, 24, 3, 2, 3, nothing, relu),
                                         (mbconv, 5, 40, 3, 2, 3, nothing, relu),
                                         (mbconv, 5, 80, 6, 2, 3, nothing, relu),
                                         (mbconv, 3, 96, 6, 1, 2, nothing, relu),
                                         (mbconv, 5, 192, 6, 2, 4, nothing, relu),
                                         (mbconv, 3, 320, 6, 1, 1, nothing, relu),
                                     ]),
                             :A1 => (32,
                                     [
                                         (dwsep_conv_norm, 3, 16, 1, 1, relu),
                                         (mbconv, 3, 24, 6, 2, 2, nothing, relu),
                                         (mbconv, 5, 40, 3, 2, 3, 4, relu),
                                         (mbconv, 3, 80, 6, 2, 4, nothing, relu),
                                         (mbconv, 3, 112, 6, 1, 2, 4, relu),
                                         (mbconv, 5, 160, 6, 2, 3, 4, relu),
                                         (mbconv, 3, 320, 6, 1, 1, nothing, relu),
                                     ]),
                             :small => (8,
                                        [
                                            (dwsep_conv_norm, 3, 8, 1, 1, relu),
                                            (mbconv, 3, 16, 3, 2, 1, nothing, relu),
                                            (mbconv, 3, 16, 6, 2, 2, nothing, relu),
                                            (mbconv, 5, 32, 6, 2, 4, 4, relu),
                                            (mbconv, 3, 32, 6, 1, 3, 4, relu),
                                            (mbconv, 5, 88, 6, 2, 3, 4, relu),
                                            (mbconv, 3, 144, 6, 1, 1, nothing, relu),
                                        ]))

function mnasnet(config::Symbol; width_mult::Real = 1, max_width::Integer = 1280,
                 dropout_prob = 0.2, inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, keys(MNASNET_CONFIGS))
    # momentum used for BatchNorm is as per Tensorflow implementation
    norm_layer = (args...; kwargs...) -> BatchNorm(args...; momentum = 0.0003f0, kwargs...)
    inplanes, block_configs = MNASNET_CONFIGS[config]
    return build_irmodel(width_mult, block_configs; inplanes, norm_layer,
                         headplanes = max_width, dropout_prob, inchannels, nclasses)
end

"""
    MNASNet(width_mult = 1; inchannels::Integer = 3, pretrain::Bool = false,
            nclasses::Integer = 1000)

Creates a MNASNet model with the specified configuration.
([reference](https://arxiv.org/abs/1807.11626))

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes

!!! warning
    
    `MNASNet` does not currently support pretrained weights.

See also [`mnasnet`](#).
"""
struct MNASNet
    layers::Any
end
@functor MNASNet

function MNASNet(config::Symbol; width_mult::Real = 1, pretrain::Bool = false,
                 inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, keys(MNASNET_CONFIGS))
    layers = mnasnet(config; width_mult, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "mnasnet$(width_mult)")
    end
    return MNASNet(layers)
end

(m::MNASNet)(x) = m.layers(x)

backbone(m::MNASNet) = m.layers[1]
classifier(m::MNASNet) = m.layers[2]
