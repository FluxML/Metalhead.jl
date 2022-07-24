"""
    mobilenetv2(width_mult, configs; inchannels = 3, max_width = 1280, nclasses = 1000)

Create a MobileNetv2 model.
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1.0 being the default in the paper)

  - `configs`: A "list of tuples" configuration for each layer that details:
    
      + `t`: The expansion factor that controls the number of feature maps in the bottleneck layer
      + `c`: The number of output feature maps
      + `n`: The number of times a block is repeated
      + `s`: The stride of the convolutional kernel
      + `a`: The activation function used in the bottleneck layer
  - `inchannels`: The number of input channels.
  - `max_width`: The maximum number of feature maps in any layer of the network
  - `nclasses`: The number of output classes
"""
function mobilenetv2(width_mult, configs; inchannels = 3, max_width = 1280, nclasses = 1000)
    # building first layer
    inplanes = _round_channels(32 * width_mult, width_mult == 0.1 ? 4 : 8)
    layers = []
    append!(layers, conv_norm((3, 3), inchannels, inplanes; pad = 1, stride = 2))
    # building inverted residual blocks
    for (t, c, n, s, a) in configs
        outplanes = _round_channels(c * width_mult, width_mult == 0.1 ? 4 : 8)
        for i in 1:n
            push!(layers,
                  invertedresidual(3, inplanes, inplanes * t, outplanes, a;
                                   stride = i == 1 ? s : 1))
            inplanes = outplanes
        end
    end
    # building last several layers
    outplanes = (width_mult > 1) ?
                _round_channels(max_width * width_mult, width_mult == 0.1 ? 4 : 8) :
                max_width
    return Chain(Chain(Chain(layers),
                       conv_norm((1, 1), inplanes, outplanes, relu6; bias = false)...),
                 Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten,
                       Dense(outplanes, nclasses)))
end

# Layer configurations for MobileNetv2
const mobilenetv2_configs = [
    #  t,   c, n, s,     a
    (1, 16, 1, 1, relu6),
    (6, 24, 2, 2, relu6),
    (6, 32, 3, 2, relu6),
    (6, 64, 4, 2, relu6),
    (6, 96, 3, 1, relu6),
    (6, 160, 3, 2, relu6),
    (6, 320, 1, 1, relu6),
]

# Model definition for MobileNetv2
struct MobileNetv2
    layers::Any
end
@functor MobileNetv2

"""
    MobileNetv2(width_mult = 1.0; inchannels = 3, pretrain = false, nclasses = 1000)

Create a MobileNetv2 model with the specified configuration.
([reference](https://arxiv.org/abs/1801.04381)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet.

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1.0 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `inchannels`: The number of input channels.
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `nclasses`: The number of output classes

See also [`Metalhead.mobilenetv2`](#).
"""
function MobileNetv2(width_mult::Number = 1; inchannels = 3, pretrain = false,
                     nclasses = 1000)
    layers = mobilenetv2(width_mult, mobilenetv2_configs; inchannels, nclasses)
    pretrain && loadpretrain!(layers, string("MobileNetv2"))
    if pretrain
        loadpretrain!(layers, string("MobileNetv2"))
    end
    return MobileNetv2(layers)
end

(m::MobileNetv2)(x) = m.layers(x)

backbone(m::MobileNetv2) = m.layers[1]
classifier(m::MobileNetv2) = m.layers[2]
