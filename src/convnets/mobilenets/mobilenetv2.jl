"""
    mobilenetv2(configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
                max_width::Integer = 1280, divisor::Integer = 8, dropout_rate = 0.2,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv2 model.
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments

  - `configs`: A "list of tuples" configuration for each layer that details:
    
      + `t`: The expansion factor that controls the number of feature maps in the bottleneck layer
      + `c`: The number of output feature maps
      + `n`: The number of times a block is repeated
      + `s`: The stride of the convolutional kernel
      + `a`: The activation function used in the bottleneck layer

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper)
  - `max_width`: The maximum number of feature maps in any layer of the network
  - `divisor`: The divisor used to round the number of feature maps in each block
  - `dropout_rate`: rate of dropout in the classifier head. Set to `nothing` to disable dropout.
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes
"""
function mobilenetv2(configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
                     max_width::Integer = 1280, divisor::Integer = 8, dropout_rate = 0.2,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    # building first layer
    inplanes = _round_channels(32 * width_mult, divisor)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes; pad = 1, stride = 2))
    # building inverted residual blocks
    for (t, c, n, s, activation) in configs
        outplanes = _round_channels(c * width_mult, divisor)
        for i in 1:n
            stride = i == 1 ? s : 1
            push!(layers,
                  mbconv((3, 3), inplanes, round(Int, inplanes * t), outplanes,
                         activation; stride))
            inplanes = outplanes
        end
    end
    # building last layers
    outplanes = _round_channels(max_width * max(1, width_mult), divisor)
    append!(layers, conv_norm((1, 1), inplanes, outplanes, relu6))
    return Chain(Chain(layers...), create_classifier(outplanes, nclasses; dropout_rate))
end

# Layer configurations for MobileNetv2
const MOBILENETV2_CONFIGS = [
    # t, c, n, s, a
    (1, 16, 1, 1, relu6),
    (6, 24, 2, 2, relu6),
    (6, 32, 3, 2, relu6),
    (6, 64, 4, 2, relu6),
    (6, 96, 3, 1, relu6),
    (6, 160, 3, 2, relu6),
    (6, 320, 1, 1, relu6),
]

"""
    MobileNetv2(width_mult = 1.0; inchannels::Integer = 3, pretrain::Bool = false,
                nclasses::Integer = 1000)

Create a MobileNetv2 model with the specified configuration.
([reference](https://arxiv.org/abs/1801.04381)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet.

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes

See also [`Metalhead.mobilenetv2`](#).
"""
struct MobileNetv2
    layers::Any
end
@functor MobileNetv2

function MobileNetv2(width_mult::Real = 1; pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = mobilenetv2(MOBILENETV2_CONFIGS; width_mult, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv2"))
    end
    return MobileNetv2(layers)
end

(m::MobileNetv2)(x) = m.layers(x)

backbone(m::MobileNetv2) = m.layers[1]
classifier(m::MobileNetv2) = m.layers[2]
