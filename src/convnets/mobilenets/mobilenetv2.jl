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

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper)
  - `max_width`: The maximum number of feature maps in any layer of the network
  - `divisor`: The divisor used to round the number of feature maps in each block
  - `dropout_rate`: rate of dropout in the classifier head. Set to `nothing` to disable dropout.
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes
"""
function mobilenetv2(block_configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
                     max_width::Integer = 1280, divisor::Integer = 8,
                     inplanes::Integer = 32, dropout_rate = 0.2,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    # building first layer
    inplanes = _round_channels(inplanes * width_mult, divisor)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes; pad = 1, stride = 2))
    # building inverted residual blocks
    get_layers, block_repeats = mbconv_stack_builder(block_configs,
                                                     fill(mbconv_builder,
                                                          length(block_configs));
                                                     inplanes)
    append!(layers, cnn_stages(get_layers, block_repeats, +))
    # building last layers
    outplanes = _round_channels(max_width * max(1, width_mult), divisor)
    append!(layers,
            conv_norm((1, 1), _round_channels(block_configs[end][2], 8),
                      outplanes, relu6))
    return Chain(Chain(layers...), create_classifier(outplanes, nclasses; dropout_rate))
end

# Layer configurations for MobileNetv2
const MOBILENETV2_CONFIGS = [
    # k, c, e, s, n, r, a
    (3, 16, 1, 1, 1, nothing, relu6),
    (3, 24, 6, 2, 2, nothing, relu6),
    (3, 32, 6, 2, 3, nothing, relu6),
    (3, 64, 6, 2, 4, nothing, relu6),
    (3, 96, 6, 1, 3, nothing, relu6),
    (3, 160, 6, 2, 3, nothing, relu6),
    (3, 320, 6, 1, 1, nothing, relu6),
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
