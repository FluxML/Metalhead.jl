# Layer configurations for small and large models for MobileNetv3
# f: mbconv block function - we use `mbconv` for all blocks
# k: kernel size
# c: output channels
# e: expansion factor
# s: stride
# n: number of repeats
# r: squeeze and excite reduction factor
# a: activation function
# Data is organised as (f, k, c, e, s, n, r, a)
const MOBILENETV3_CONFIGS = Dict(:small => (1024,
                                            [
                                                (mbconv, 3, 16, 1, 2, 1, 4, relu),
                                                (mbconv, 3, 24, 4.5, 2, 1, nothing, relu),
                                                (mbconv, 3, 24, 3.67, 1, 1, nothing, relu),
                                                (mbconv, 5, 40, 4, 2, 1, 4, hardswish),
                                                (mbconv, 5, 40, 6, 1, 2, 4, hardswish),
                                                (mbconv, 5, 48, 3, 1, 2, 4, hardswish),
                                                (mbconv, 5, 96, 6, 1, 3, 4, hardswish),
                                            ]),
                                 :large => (1280,
                                            [
                                                (mbconv, 3, 16, 1, 1, 1, nothing, relu),
                                                (mbconv, 3, 24, 4, 2, 1, nothing, relu),
                                                (mbconv, 3, 24, 3, 1, 1, nothing, relu),
                                                (mbconv, 5, 40, 3, 2, 3, 4, relu),
                                                (mbconv, 3, 80, 6, 2, 1, nothing,
                                                 hardswish),
                                                (mbconv, 3, 80, 2.5, 1, 1, nothing,
                                                 hardswish),
                                                (mbconv, 3, 80, 2.3, 1, 2, nothing,
                                                 hardswish),
                                                (mbconv, 3, 112, 6, 1, 2, 4, hardswish),
                                                (mbconv, 5, 160, 6, 1, 3, 4, hardswish),
                                            ]))

function mobilenetv3(config::Symbol; width_mult::Real = 1, dropout_prob = 0.2,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, [:small, :large])
    max_width, block_configs = MOBILENETV3_CONFIGS[config]
    return irmodelbuilder(width_mult, block_configs; inplanes = 16,
                          headplanes = max_width, activation = relu,
                          se_from_explanes = true, se_round_fn = _round_channels,
                          expanded_classifier = true, dropout_prob, inchannels, nclasses)
end

"""
    MobileNetv3(config::Symbol; width_mult::Real = 1, pretrain::Bool = false,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv3 model with the specified configuration.
([reference](https://arxiv.org/abs/1905.02244)).
Set `pretrain = true` to load the model with pre-trained weights for ImageNet.

# Arguments

  - `config`: :small or :large for the size of the model (see paper).
  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `pretrain`: whether to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels
  - `nclasses`: the number of output classes

!!! warning
    
    `MobileNetv3` does not currently support pretrained weights.

See also [`mobilenetv3`](#).
"""
struct MobileNetv3
    layers::Any
end
@functor MobileNetv3

function MobileNetv3(config::Symbol; width_mult::Real = 1, pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = mobilenetv3(config; width_mult, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv3", config))
    end
    return MobileNetv3(layers)
end

(m::MobileNetv3)(x) = m.layers(x)

backbone(m::MobileNetv3) = m.layers[1]
classifier(m::MobileNetv3) = m.layers[2]
