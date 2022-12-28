# Layer configurations for MobileNetv2
# f: block function - we use `mbconv` for all blocks
# k: kernel size
# c: output channels
# e: expansion factor
# s: stride
# n: number of repeats
# r: reduction factor
# a: activation function
# Data is organised as (f, k, c, e, s, n, r, a)
const MOBILENETV2_CONFIGS = [
    (mbconv, 3, 16, 1, 1, 1, nothing, relu6),
    (mbconv, 3, 24, 6, 2, 2, nothing, relu6),
    (mbconv, 3, 32, 6, 2, 3, nothing, relu6),
    (mbconv, 3, 64, 6, 2, 4, nothing, relu6),
    (mbconv, 3, 96, 6, 1, 3, nothing, relu6),
    (mbconv, 3, 160, 6, 2, 3, nothing, relu6),
    (mbconv, 3, 320, 6, 1, 1, nothing, relu6),
]

"""
    mobilenetv2(width_mult::Real = 1; max_width::Integer = 1280,
                inplanes::Integer = 32, dropout_prob = 0.2,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv2 model. ([reference](https://arxiv.org/abs/1801.04381v1)).

# Arguments

    - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper; this is usually a value between 0.1 and 1.4)
    - `max_width`: The maximum width of the network.
    - `inplanes`: Number of input channels to the first convolution layer
    - `dropout_prob`: Dropout probability for the classifier head. Set to `nothing` to disable dropout.
    - `inchannels`: Number of input channels.
    - `nclasses`: Number of output classes.
"""
function mobilenetv2(width_mult::Real = 1; max_width::Integer = 1280,
                     inplanes::Integer = 32, dropout_prob = 0.2,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    return build_invresmodel(width_mult, MOBILENETV2_CONFIGS; activation = relu6, inplanes,
                             headplanes = max_width, dropout_prob, inchannels, nclasses)
end

"""
    MobileNetv2(width_mult = 1.0; inchannels::Integer = 3, pretrain::Bool = false,
                nclasses::Integer = 1000)

Create a MobileNetv2 model with the specified configuration.
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper; this is usually a value between 0.1 and 1.4)
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes

!!! warning

    `MobileNetv2` does not currently support pretrained weights.

See also [`Metalhead.mobilenetv2`](@ref).
"""
struct MobileNetv2
    layers::Any
end
@functor MobileNetv2

function MobileNetv2(width_mult::Real = 1; pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = mobilenetv2(width_mult; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv2"))
    end
    return MobileNetv2(layers)
end

(m::MobileNetv2)(x) = m.layers(x)

backbone(m::MobileNetv2) = m.layers[1]
classifier(m::MobileNetv2) = m.layers[2]
