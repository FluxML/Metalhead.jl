# This is a utility function for making sure that all layers have a channel size divisible by 8.
function _make_divisible(v, divisor, min_value = nothing)
  if isnothing(min_value)
    min_value = divisor
  end
  new_v = max(min_value, floor(Int, v + divisor / 2) ÷ divisor * divisor)
  # Make sure that round down does not go down by more than 10%
  (new_v < 0.9 * v) ? new_v + divisor : new_v
end

# MobileNetv2

"""
    invertedresidualv2(inplanes, outplanes, stride, expand_ratio)

Create a basic inverted residual block for MobileNetv2
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments
- `inplanes`: The number of input feature maps
- `outplanes`: The number of output feature maps
- `stride`: The stride of the convolutional layer, has to be either 1 or 2
- `expand_ratio`: The ratio of the inner bottleneck feature maps over the input feature maps
"""
function invertedresidualv2(inplanes, outplanes, stride, expand_ratio)
  @assert stride in [1, 2] "`stride` has to be 1 or 2"
  hidden_planes = floor(Int, inplanes * expand_ratio)

  if expand_ratio == 1
    invres = Chain(conv_bn((3, 3), hidden_planes, hidden_planes, relu6;
                           bias = false, stride, pad = 1, groups = hidden_planes)...,
                   conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...)
  else
    invres = Chain(conv_bn((1, 1), inplanes, hidden_planes, relu6; bias = false)...,
                   conv_bn((3, 3), hidden_planes, hidden_planes, relu6;
                           bias = false, stride, pad = 1, groups = hidden_planes)...,
                   conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...)
  end

  (stride == 1 && inplanes == outplanes) ? SkipConnection(invres, +) : invres
end

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
- `max_width`: The maximum number of feature maps in any layer of the network
- `nclasses`: The number of output classes
"""
function mobilenetv2(width_mult, configs; max_width = 1280, nclasses = 1000)
  # building first layer
  inplanes = _make_divisible(32 * width_mult, width_mult == 0.1 ? 4 : 8)
  layers = []
  append!(layers, conv_bn((3, 3), 3, inplanes, stride = 2))

  # building inverted residual blocks
  for (t, c, n, s) in configs
    outplanes = _make_divisible(c * width_mult, width_mult == 0.1 ? 4 : 8)
    for i in 1:n
      push!(layers, invertedresidualv2(inplanes, outplanes, i == 1 ? s : 1, t))
      inplanes = outplanes
    end
  end

  # building last several layers
  outplanes = (width_mult > 1.0) ? _make_divisible(max_width * width_mult, width_mult == 0.1 ? 4 : 8) : max_width

  return Chain(Chain(layers...,
                     conv_bn((1, 1), inplanes, outplanes, relu6, bias = false)...),
               Chain(AdaptiveMeanPool((1, 1)), flatten, Dense(outplanes, nclasses)))
end

# Layer configurations for MobileNetv2
const mobilenetv2_configs = [
  # t, c, n, s
  (1, 16, 1, 1),
  (6, 24, 2, 2),
  (6, 32, 3, 2),
  (6, 64, 4, 2),
  (6, 96, 3, 1),
  (6, 160, 3, 2),
  (6, 320, 1, 1)
]

# Model definition for MobileNetv2
struct MobileNetv2
  layers
end

"""
    MobileNetv2(width_mult::Number = 1; pretrain = false, nclasses = 1000)

Create a MobileNetv2 model with the specified configuration.
([reference](https://arxiv.org/abs/1801.04381)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet. 

# Arguments
- `width_mult`: Controls the number of feature maps in each layer, with 1.0 being the original
  model as detailed in the paper. This is usually a floating point value in between 0.1 and 1.4.
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
    selayer(channels, reduction = 4)

Squeeze and Excitation layer used by MobileNetv3
([reference](https://arxiv.org/abs/1905.02244)).
"""
selayer(channels, reduction = 4) =
  SkipConnection(Chain(AdaptiveMeanPool((1, 1)),
                       conv_bn((1, 1), channels, channels // reduction, relu; bias = false)...,
                       conv_bn((1, 1), channels // reduction, channels, hardσ)...,), .*)

"""
    invertedresidualv3(inplanes, hidden_planes, outplanes, kernel_size, stride, use_se, use_hs)

Create a basic inverted residual block for MobileNetv3
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `inplanes`: The number of input feature maps
- `hidden_planes`: The number of feature maps in the hidden layer
- `outplanes`: The number of output feature maps
- `kernel_size`: The kernel size of the convolutional layers
- `stride`: The stride of the convolutional kernel, has to be either 1 or 2
- `use_se`: If `true`, Squeeze and Excitation layer will be used
- `use_hs`: If `true`, Hard-Swish activation function will be used
"""
function invertedresidualv3(inplanes, hidden_planes, outplanes, kernel_size,
                            stride, use_se, use_hs)
  @assert stride in [1, 2] "`stride` has to be 1 or 2"

  if inplanes == hidden_planes
    invres = Chain(conv_bn((kernel_size, kernel_size), hidden_planes, hidden_planes, use_hs ? hardswish : relu;
                            bias = false, stride, pad = (kernel_size - 1) ÷ 2, groups = hidden_planes)...,
                            use_se ? selayer(hidden_planes) : identity,
                   conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...)
  else
    invres = Chain(conv_bn((1, 1), inplanes, hidden_planes, use_hs ? hardswish : relu; bias = false)...,
                   conv_bn((kernel_size, kernel_size), hidden_planes, hidden_planes, use_hs ? hardswish : relu;
                            bias = false, stride, pad = (kernel_size - 1) ÷ 2, groups = hidden_planes)...,
                            use_se ? selayer(hidden_planes) : identity,
                   conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...)
  end

  (stride == 1 && inplanes == outplanes) ? SkipConnection(invres, +) : invres
end

"""
    mobilenetv3(width_mult, configs; max_width = 1024, nclasses = 1000)

Create a MobileNetv3 model.
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `configs`: a "list of tuples" configuration for each layer that details:
  - `k::Int` - The size of the convolutional kernel
  - `c::Float` - The multiplier factor for deciding the number of feature maps in the hidden layer
  - `t::Int` - The number of output feature maps for a given block
  - `use_se::Bool` - Whether to use Squeeze and Excitation layer
  - `use_hs::Bool` - Whether to use Hard-Swish activation function
  - `s::Int` - The stride of the convolutional kernel
- `width_mult`: Controls the number of feature maps in each layer, with 1.0 being the original
  model as detailed in the paper.
- `max_width`: The maximum number of feature maps in any layer of the network
- `nclasses`: the number of output classes
"""
function mobilenetv3(width_mult, configs; max_width = 1024, nclasses = 1000)
  # building first layer
  inplanes = _make_divisible(16 * width_mult, 8)
  layers = []
  append!(layers, conv_bn((3, 3), 3, inplanes, hardswish; stride = 2))
  explanes = 0
  # building inverted residual blocks
  for (k, t, c, use_se, use_hs, s) in configs
    # inverted residual layers
    outplanes = _make_divisible(c * width_mult, 8)
    explanes = _make_divisible(inplanes * t, 8)
    push!(layers, invertedresidualv3(inplanes, explanes, outplanes, k, s, use_se, use_hs))
    inplanes = outplanes
  end

  # building last several layers
  output_channel = max_width
  output_channel = width_mult > 1.0 ? _make_divisible(output_channel * width_mult, 8) : output_channel
  classifier = (
    Dense(explanes, output_channel, hardswish),
    Dropout(0.2),
    Dense(output_channel, nclasses),
  )

  return Chain(Chain(layers...,
                     conv_bn((1, 1), inplanes, explanes, hardswish, bias = false)...),
               Chain(AdaptiveMeanPool((1, 1)), flatten, classifier...))
end

# Configurations for small and large mode for MobileNetv3
mobilenetv3_configs = Dict(
  :small => [
    # k, t, c, SE, HS, s 
    (3, 1, 16, true, false, 2),
    (3, 4.5, 24, false, false, 2),
    (3, 3.67, 24, false, false, 1),
    (5, 4, 40, true, true, 2),
    (5, 6, 40, true, true, 1),
    (5, 6, 40, true, true, 1),
    (5, 3, 48, true, true, 1),
    (5, 3, 48, true, true, 1),
    (5, 6, 96, true, true, 2),
    (5, 6, 96, true, true, 1),
    (5, 6, 96, true, true, 1),
  ], 
  :large => [
    # k, t, c, SE, HS, s 
    (3, 1, 16, false, false, 1),
    (3, 4, 24, false, false, 2),
    (3, 3, 24, false, false, 1),
    (5, 3, 40, true, false, 2),
    (5, 3, 40, true, false, 1),
    (5, 3, 40, true, false, 1),
    (3, 6, 80, false, true, 2),
    (3, 2.5, 80, false, true, 1),
    (3, 2.3, 80, false, true, 1),
    (3, 2.3, 80, false, true, 1),
    (3, 6, 112, true, true, 1),
    (3, 6, 112, true, true, 1),
    (5, 6, 160, true, true, 2),
    (5, 6, 160, true, true, 1),
    (5, 6, 160, true, true, 1)
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
- `width_mult`: Controls the number of output feature maps in each block (with 1.0 being the 
  default in the paper). This is usually a floating point value in between 0.1 and 1.4.
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
