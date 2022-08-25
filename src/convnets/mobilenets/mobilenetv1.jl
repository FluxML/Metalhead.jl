# Layer configurations for MobileNetv1
# f: block function - we use `dwsep_conv_norm` for all blocks
# k: kernel size
# c: output channels
# s: stride
# n: number of repeats
# a: activation function
# Data is organised as (f, k, c, s, n, a)
const MOBILENETV1_CONFIGS = [
    (dwsep_conv_norm, 3, 64, 1, 1, relu6),
    (dwsep_conv_norm, 3, 128, 2, 2, relu6),
    (dwsep_conv_norm, 3, 256, 2, 2, relu6),
    (dwsep_conv_norm, 3, 512, 2, 6, relu6),
    (dwsep_conv_norm, 3, 1024, 2, 2, relu6),
]

function mobilenetv1(width_mult::Real = 1; inplanes::Integer = 32, dropout_rate = nothing,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    return irmodelbuilder(width_mult, MOBILENETV1_CONFIGS; inplanes, inchannels,
                          activation = relu6, connection = nothing, tail_conv = false,
                          headplanes = 1024, dropout_rate, nclasses)
end

"""
    MobileNetv1(width_mult::Real = 1; pretrain::Bool = false,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv1 model with the baseline configuration
([reference](https://arxiv.org/abs/1704.04861v1)).

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes

!!! warning
    
    `MobileNetv1` does not currently support pretrained weights.

See also [`mobilenetv1`](#).
"""
struct MobileNetv1
    layers::Any
end
@functor MobileNetv1

function MobileNetv1(width_mult::Real = 1; pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = mobilenetv1(width_mult; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv1"))
    end
    return MobileNetv1(layers)
end

(m::MobileNetv1)(x) = m.layers(x)

backbone(m::MobileNetv1) = m.layers[1]
classifier(m::MobileNetv1) = m.layers[2]
