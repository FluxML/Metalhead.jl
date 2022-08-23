# momentum used for BatchNorm as per Tensorflow implementation
const _MNASNET_BN_MOMENTUM = 0.0003f0

"""
    mnasnet(block_configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
            max_width = 1280, dropout_rate = 0.2, inchannels::Integer = 3,
            nclasses::Integer = 1000)

Create an MNASNet model with the specified configuration.
([reference](https://arxiv.org/abs/1807.11626)).

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper)
  - `max_width`: The maximum number of feature maps in any layer of the network
  - `dropout_rate`: rate of dropout in the classifier head. Set to `nothing` to disable dropout.
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes
"""
function mnasnet(block_configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
                 max_width::Integer = 1280, inplanes::Integer = 32, dropout_rate = 0.2,
                 inchannels::Integer = 3, nclasses::Integer = 1000)
    # norm layer for MNASNet is different from other models
    norm_layer = (args...; kwargs...) -> BatchNorm(args...; momentum = _MNASNET_BN_MOMENTUM,
                                                   kwargs...)
    # building first layer
    inplanes = _round_channels(inplanes * width_mult)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes, relu; stride = 2, pad = 1,
                      norm_layer))
    # building inverted residual blocks
    get_layers, block_repeats = mbconv_stack_builder(block_configs, inplanes; width_mult,
                                                     norm_layer)
    append!(layers, cnn_stages(get_layers, block_repeats, +))
    # building last layers
    outplanes = _round_channels(block_configs[end][3] * width_mult)
    append!(layers,
            conv_norm((1, 1), outplanes, max_width, relu; norm_layer))
    return Chain(Chain(layers...), create_classifier(max_width, nclasses; dropout_rate))
end

# Layer configurations for MNasNet
# f: block function - we use `dwsep_conv_bn` for the first block and `mbconv` for the rest
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
                                         (dwsep_conv_bn, 3, 16, 1, 1, relu),
                                         (mbconv, 3, 24, 3, 2, 3, nothing, relu),
                                         (mbconv, 5, 40, 3, 2, 3, nothing, relu),
                                         (mbconv, 5, 80, 6, 2, 3, nothing, relu),
                                         (mbconv, 3, 96, 6, 1, 2, nothing, relu),
                                         (mbconv, 5, 192, 6, 2, 4, nothing, relu),
                                         (mbconv, 3, 320, 6, 1, 1, nothing, relu),
                                     ]),
                             :A1 => (32,
                                     [
                                         (dwsep_conv_bn, 3, 16, 1, 1, relu),
                                         (mbconv, 3, 24, 6, 2, 2, nothing, relu),
                                         (mbconv, 5, 40, 3, 2, 3, 4, relu),
                                         (mbconv, 3, 80, 6, 2, 4, nothing, relu),
                                         (mbconv, 3, 112, 6, 1, 2, 4, relu),
                                         (mbconv, 5, 160, 6, 2, 3, 4, relu),
                                         (mbconv, 3, 320, 6, 1, 1, nothing, relu),
                                     ]),
                             :small => (8,
                                        [
                                            (dwsep_conv_bn, 3, 8, 1, 1, relu),
                                            (mbconv, 3, 16, 3, 2, 1, nothing, relu),
                                            (mbconv, 3, 16, 6, 2, 2, nothing, relu),
                                            (mbconv, 5, 32, 6, 2, 4, 4, relu),
                                            (mbconv, 3, 32, 6, 1, 3, 4, relu),
                                            (mbconv, 5, 88, 6, 2, 3, 4, relu),
                                            (mbconv, 3, 144, 6, 1, 1, nothing, relu)]))

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
    inplanes, block_configs = MNASNET_CONFIGS[config]
    layers = mnasnet(block_configs; width_mult, inplanes, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "mnasnet$(width_mult)")
    end
    return MNASNet(layers)
end

(m::MNASNet)(x) = m.layers(x)

backbone(m::MNASNet) = m.layers[1]
classifier(m::MNASNet) = m.layers[2]
