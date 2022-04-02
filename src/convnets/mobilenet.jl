# MobileNetv1

"""
    mobilenetv1(imsize, width_mult, config;
                activation = relu,
                inchannels = 3,
                nclasses = 1000,
                fcsize = 1024)

Create a MobileNetv1 model ([reference](https://arxiv.org/abs/1704.04861v1)).

# Arguments
- `imsize`: A 2-tuple indicating the input spatial dimensions
- `width_mult`: Controls the number of output feature maps in each block
                (with 1.0 being the default in the paper)
- `configs`: A "list of tuples" configuration for each layer that details:
  - `dw`: Set true to use a depthwise separable convolution or false for regular convolution
  - `o`: The number of output feature maps
  - `s`: The stride of the convolutional kernel
  - `r`: The number of time this configuration block is repeated
- `fcsize`: The intermediate fully-connected size between the convolution and final layers
- `nclasses`: The number of output classes
"""
function mobilenetv1(imsize, width_mult, config;
                     activation = relu,
                     inchannels = 3,
                     nclasses = 1000,
                     fcsize = 1024)
  layers = []
  inch = inchannels
  for (dw, outch, stride, repeats) in config
    for _ in 1:repeats
      outch = outch * width_mult
      layer = if dw
        depthwise_sep_conv_bn((3, 3), inch, outch, activation; stride = stride, pad = 1)
      else
        conv_bn((3, 3), inch, outch, activation; stride = stride, pad = 1)
      end
      append!(layers, layer)
      inch = outch
    end
  end
  push!(layers, MeanPool((7, 7)))
  convsize = prod(Flux.outputsize(layers, (imsize..., inchannels, 1)))

  return Chain(Chain(layers...),
               Chain(MLUtils.flatten,
                     Dense(convsize, fcsize, activation),
                     Dense(fcsize, nclasses)))
end

const mobilenetv1_configs = [
#     dw,    c, s, r
  (false,   32, 2, 1),
  ( true,   64, 1, 1),
  ( true,  128, 2, 1),
  ( true,  128, 1, 1),
  ( true,  256, 2, 1),
  ( true,  256, 1, 1),
  ( true,  512, 2, 1),
  ( true,  512, 1, 5),
  ( true, 1024, 2, 1),
  ( true, 1024, 1, 1)
]

"""
    MobileNetv1(imsize::Dims{2} = (224, 224), width_mult = 1;
                pretrain = false, nclasses = 1000)

Create a MobileNetv1 model with the baseline configuration 
([reference](https://arxiv.org/abs/1704.04861v1)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet.

# Arguments
- `imsize`: A 2-tuple indicating the input spatial dimensions
- `width_mult`: Controls the number of output feature maps in each block
                (with 1.0 being the default in the paper;
                 this is usually a value between 0.1 and 1.4)
- `pretrain`: Whether to load the pre-trained weights for ImageNet
- `nclasses`: The number of output classes

See also [`Metalhead.mobilenetv1`](#).
"""
struct MobileNetv1
  layers
end

function MobileNetv1(imsize::Dims{2} = (224, 224), width_mult = 1;
                     pretrain = false, nclasses = 1000)
  layers = mobilenetv1(imsize, width_mult, mobilenetv1_configs; nclasses = nclasses)
  pretrain && loadpretrain!(layers, string("MobileNetv1"))

  return MobileNetv1(layers)
end

@functor MobileNetv1

(m::MobileNetv1)(x) = m.layers(x)

backbone(m::MobileNetv1) = m.layers[1]
classifier(m::MobileNetv1) = m.layers[2]

# MobileNetv2

"""
    mobilenetv2(width_mult, configs; max_width = 1280, nclasses = 1000)

Create a MobileNetv2 model.
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments
- `width_mult`: Controls the number of output feature maps in each block
                (with 1.0 being the default in the paper)
- `configs`: A "list of tuples" configuration for each layer that details:
  - `t`: The expansion factor that controls the number of feature maps in the bottleneck layer
  - `c`: The number of output feature maps
  - `n`: The number of times a block is repeated
  - `s`: The stride of the convolutional kernel
  - `a`: The activation function used in the bottleneck layer
- `max_width`: The maximum number of feature maps in any layer of the network
- `nclasses`: The number of output classes
"""
function mobilenetv2(width_mult, configs; max_width = 1280, nclasses = 1000)
  # building first layer
  inplanes = _round_channels(32 * width_mult, width_mult == 0.1 ? 4 : 8)
  layers = []
  append!(layers, conv_bn((3, 3), 3, inplanes, stride = 2))

  # building inverted residual blocks
  for (t, c, n, s, a) in configs
    outplanes = _round_channels(c * width_mult, width_mult == 0.1 ? 4 : 8)
    for i in 1:n
      push!(layers, invertedresidual(3, inplanes, inplanes * t, outplanes, a;
                                     stride = i == 1 ? s : 1))
      inplanes = outplanes
    end
  end

  # building last several layers
  outplanes = (width_mult > 1) ? _round_channels(max_width * width_mult, width_mult == 0.1 ? 4 : 8) :
                                 max_width

  return Chain(Chain(layers...,
                     conv_bn((1, 1), inplanes, outplanes, relu6, bias = false)...),
               Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten, Dense(outplanes, nclasses)))
end

# Layer configurations for MobileNetv2
const mobilenetv2_configs = [
#  t,   c, n, s,     a
  (1,  16, 1, 1, relu6),
  (6,  24, 2, 2, relu6),
  (6,  32, 3, 2, relu6),
  (6,  64, 4, 2, relu6),
  (6,  96, 3, 1, relu6),
  (6, 160, 3, 2, relu6),
  (6, 320, 1, 1, relu6)
]

# Model definition for MobileNetv2
struct MobileNetv2
  layers
end

"""
    MobileNetv2(width_mult = 1.0; pretrain = false, nclasses = 1000)

Create a MobileNetv2 model with the specified configuration.
([reference](https://arxiv.org/abs/1801.04381)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet. 

# Arguments
- `width_mult`: Controls the number of output feature maps in each block
                (with 1.0 being the default in the paper;
                 this is usually a value between 0.1 and 1.4)
- `pretrain`: Whether to load the pre-trained weights for ImageNet
- `nclasses`: The number of output classes

See also [`Metalhead.mobilenetv2`](#).
"""
function MobileNetv2(width_mult::Number = 1; pretrain = false, nclasses = 1000)
  layers = mobilenetv2(width_mult, mobilenetv2_configs; nclasses = nclasses)
  pretrain && loadpretrain!(layers, string("MobileNetv2"))

  MobileNetv2(layers)
end

@functor MobileNetv2

(m::MobileNetv2)(x) = m.layers(x)

backbone(m::MobileNetv2) = m.layers[1]
classifier(m::MobileNetv2) = m.layers[2:end]

# MobileNetv3

"""
    mobilenetv3(width_mult, configs; max_width = 1024, nclasses = 1000)

Create a MobileNetv3 model.
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `width_mult`: Controls the number of output feature maps in each block
                (with 1.0 being the default in the paper;
                 this is usually a value between 0.1 and 1.4)
- `configs`: a "list of tuples" configuration for each layer that details:
  - `k::Int` - The size of the convolutional kernel
  - `c::Float` - The multiplier factor for deciding the number of feature maps in the hidden layer
  - `t::Int` - The number of output feature maps for a given block
  - `r::Int` - The reduction factor (`>= 1` or `nothing` to skip) for squeeze and excite layers
  - `s::Int` - The stride of the convolutional kernel
  - `a` - The activation function used in the bottleneck (typically `hardswish` or `relu`)
- `max_width`: The maximum number of feature maps in any layer of the network
- `nclasses`: the number of output classes
"""
function mobilenetv3(width_mult, configs; max_width = 1024, nclasses = 1000)
  # building first layer
  inplanes = _round_channels(16 * width_mult, 8)
  layers = []
  append!(layers, conv_bn((3, 3), 3, inplanes, hardswish; stride = 2))
  explanes = 0
  # building inverted residual blocks
  for (k, t, c, r, a, s) in configs
    # inverted residual layers
    outplanes = _round_channels(c * width_mult, 8)
    explanes = _round_channels(inplanes * t, 8)
    push!(layers, invertedresidual(k, inplanes, explanes, outplanes, a;
                                   stride = s, reduction = r))
    inplanes = outplanes
  end

  # building last several layers
  output_channel = max_width
  output_channel = width_mult > 1.0 ? _round_channels(output_channel * width_mult, 8) : output_channel
  classifier = (Dense(explanes, output_channel, hardswish),
                Dropout(0.2),
                Dense(output_channel, nclasses))

  return Chain(Chain(layers...,
                     conv_bn((1, 1), inplanes, explanes, hardswish, bias = false)...),
               Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten, classifier...))
end

# Configurations for small and large mode for MobileNetv3
mobilenetv3_configs = Dict(
  :small => [
  #  k,    t,  c,      SE,         a, s
    (3,    1, 16,       4,      relu, 2),
    (3,  4.5, 24, nothing,      relu, 2),
    (3, 3.67, 24, nothing,      relu, 1),
    (5,    4, 40,       4, hardswish, 2),
    (5,    6, 40,       4, hardswish, 1),
    (5,    6, 40,       4, hardswish, 1),
    (5,    3, 48,       4, hardswish, 1),
    (5,    3, 48,       4, hardswish, 1),
    (5,    6, 96,       4, hardswish, 2),
    (5,    6, 96,       4, hardswish, 1),
    (5,    6, 96,       4, hardswish, 1),
  ], 
  :large => [
  #  k,   t,   c,      SE,         a, s
    (3,   1,  16, nothing,      relu, 1),
    (3,   4,  24, nothing,      relu, 2),
    (3,   3,  24, nothing,      relu, 1),
    (5,   3,  40,       4,      relu, 2),
    (5,   3,  40,       4,      relu, 1),
    (5,   3,  40,       4,      relu, 1),
    (3,   6,  80, nothing, hardswish, 2),
    (3, 2.5,  80, nothing, hardswish, 1),
    (3, 2.3,  80, nothing, hardswish, 1),
    (3, 2.3,  80, nothing, hardswish, 1),
    (3,   6, 112,       4, hardswish, 1),
    (3,   6, 112,       4, hardswish, 1),
    (5,   6, 160,       4, hardswish, 2),
    (5,   6, 160,       4, hardswish, 1),
    (5,   6, 160,       4, hardswish, 1)
  ]
)

# Model definition for MobileNetv3
struct MobileNetv3
  layers
end

"""
    MobileNetv3(mode::Symbol = :small, width_mult::Number = 1; pretrain = false, nclasses = 1000)

Create a MobileNetv3 model with the specified configuration.
([reference](https://arxiv.org/abs/1905.02244)).
Set `pretrain = true` to load the model with pre-trained weights for ImageNet.

# Arguments
- `mode`: :small or :large for the size of the model (see paper).
- `width_mult`: Controls the number of output feature maps in each block
                (with 1.0 being the default in the paper;
                 this is usually a value between 0.1 and 1.4)
- `pretrain`: whether to load the pre-trained weights for ImageNet
- `nclasses`: the number of output classes

See also [`Metalhead.mobilenetv3`](#).
"""
function MobileNetv3(mode::Symbol = :small, width_mult::Number = 1; pretrain = false, nclasses = 1000)
  @assert mode in [:large, :small] "`mode` has to be either :large or :small"

  max_width = (mode == :large) ? 1280 : 1024
  layers = mobilenetv3(width_mult, mobilenetv3_configs[mode]; max_width = max_width, nclasses = nclasses)
  pretrain && loadpretrain!(layers, string("MobileNetv3", mode))
  MobileNetv3(layers)
end

@functor MobileNetv3

(m::MobileNetv3)(x) = m.layers(x)

backbone(m::MobileNetv3) = m.layers[1]
classifier(m::MobileNetv3) = m.layers[2:end]
