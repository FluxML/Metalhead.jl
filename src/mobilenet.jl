"""
  This is a utility function for making sure that all layers have a channel size divisible by 8. 
"""
function _make_divisible(v, divisor, min_value = nothing)
  if isnothing(min_value)
    min_value = divisor
  end
  new_v = max(min_value, floor(Int, v + divisor / 2) ÷ divisor * divisor)
  # Make sure that round down does not go down by more than 10%
  (new_v < 0.9 * v) ? new_v + divisor : new_v
end

"""
    invertedresidualv2(inplanes, outplanes, stride, expand_ratio)

Create a basic inverted residual block for MobileNetv2
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments
- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `stride`: stride of the convolutional layer, has to be either 1 or 2
- `expand_ratio`: ratio of the input feature maps and the inner bottleneck feature maps
"""
function invertedresidualv2(inplanes, outplanes, stride, expand_ratio)
  @assert stride in [1, 2] "`stride` has to be 1 or 2"
  hidden_planes = floor(Int, inplanes * expand_ratio)

  if expand_ratio == 1
    invres = Chain(
      conv_bn((3, 3), hidden_planes, hidden_planes, relu6; bias = false, stride,
               pad = 1, groups = hidden_planes)...,
      conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...
    )
  else
    invres = Chain(
      conv_bn((1, 1), inplanes, hidden_planes, relu6; bias = false)...,
      conv_bn((3, 3), hidden_planes, hidden_planes, relu6; bias = false, stride,
               pad = 1, groups = hidden_planes)...,
      conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...
    )
  end

  (stride == 1 && inplanes == outplanes) ? SkipConnection(+, invres) : invres
end

"""
    mobilenetv2(width_mult; nclasses = 1000)

Create a MobileNetv2 model.
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments
- `width_mult`: Controls the number of channels in each layer, with 1.0 being the original
  model as detailed in the paper.
- `nclasses`: the number of output classes
"""
function mobilenetv2(width_mult; nclasses = 1000)
  configs = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
  ]
  # building first layer
  inplanes = _make_divisible(32 * width_mult, width_mult == 0.1 ? 4 : 8)
  layers = []
  push!(layers, conv_bn((3, 3), 3, inplanes, stride = 2))
  # building inverted residual blocks
  for (t, c, n, s) in configs
    outplanes = _make_divisible(c * width_mult, width_mult == 0.1 ? 4 : 8)
    for i in 1:n
      push!(layers, invertedresidualv2(inplanes, outplanes, i == 1 ? s : 1, t))
      inplanes = outplanes
    end
  end
  # building last several layers
  outplanes = (width_mult > 1.0) ? _make_divisible(1280 * width_mult, width_mult == 0.1 ? 4 : 8) : 1280

  return Chain(Chain(layers...),
               conv_bn((1, 1), inplanes, outplanes, relu6, bias = false),
               Chain(AdaptiveMeanPool((1, 1)), flatten, Dense(outplanes, nclasses)))
end

"""
Squeeze and Excitation layer used by MobileNetv3
([reference](https://arxiv.org/abs/1905.02244)).
"""
function selayer(channel, reduction = 4)
  Chain(
    Parallel(*, Chain(
        Dense(channel, _make_divisible(channel ÷ reduction, 8), relu),
        Dense(_make_divisible(channel ÷ reduction, 8), channel, hardσ),
        AdaptiveMeanPool((1, 1)), flatten
      ), identity
    )
  )
end

"""
Hard-Swish activation function
([reference](https://arxiv.org/abs/1905.02244)).
"""
@inline h_swish(x) = x * hardσ(x)

"""
    invertedresidualv3(inplanes, hidden_planes, outplanes, kernel_size, stride, use_se, use_hs)

Create a basic inverted residual block for MobileNetv3
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `inplanes`: number of input feature maps
- `hidden_planes`: number of feature maps in the hidden layer
- `outplanes`: number of output feature maps
- `kernel_size`: kernel size of the convolutional layers
- `stride`: stride of the convolutional kernel, has to be either 1 or 2
- `use_se`: if True, Squeeze and Excitation layer will be used
- `use_hs`: if True, Hard-Swish activation function will be used
"""
function invertedresidualv3(inplanes, hidden_planes, outplanes, kernel_size,
                            stride, use_se, use_hs)
  @assert stride in [1, 2] "`stride` has to be 1 or 2"

  if inplanes == hidden_planes
    invres = Chain(
      conv_bn((kernel_size, kernel_size), hidden_planes, hidden_planes, use_hs ? h_swish : relu;
               bias = false, stride, pad = (kernel_size - 1) ÷ 2, groups = hidden_planes)...,
      use_se ? selayer(hidden_planes) : identity,
      conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...
    )
  else
    invres = Chain(
      conv_bn((1, 1), inplanes, hidden_planes, use_hs ? h_swish : relu; bias = false)...,
      conv_bn((kernel_size, kernel_size), hidden_planes, hidden_planes, use_hs ? h_swish : relu;
               bias = false, stride, pad = (kernel_size - 1) ÷ 2, groups = hidden_planes)...,
      use_se ? selayer(hidden_planes) : identity,
      conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...
    )
  end

  (stride == 1 && inplanes == outplanes) ? SkipConnection(+, invres) : invres
end

"""
    mobilenetv3(configs, mode, width_mult; nclasses = 1000)

Create a MobileNetv3 model.
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `configs`: a list of configurations for each layer that details 
- `mode`: 'large' or 'small'
- `width_mult`: Controls the number of channels in each layer, with 1.0 being the original
  model as detailed in the paper.
- `nclasses`: the number of output classes
"""
function mobilenetv3(configs, mode, width_mult; nclasses = 1000)
  @assert mode in ["large", "small"] "`mode` has to be either \"large\" or \"small\""
  # building first layer
  inplanes = _make_divisible(16 * width_mult, 8)
  layers = []
  push!(layers, conv_bn((3, 3), 3, inplanes, h_swish; stride = 2))
  # building inverted residual blocks
  for (k, t, c, use_se, use_hs, s) in configs
    # converting the stored parameters to their respective types
    k = convert(Int, k)
    c = convert(Int, c)
    use_se = convert(Bool, use_se)
    use_hs = convert(Bool, use_hs)
    s = convert(Int, s)
    # inverted residual layers
    outplanes = _make_divisible(c * width_mult, 8)
    global explanes = _make_divisible(inplanes * t, 8)
    push!(layers, invertedresidualv3(inplanes, explanes, outplanes, k, s, use_se, use_hs))
    inplanes = outplanes
  end
  # building last several layers
  output_channel = Dict("large" => 1280, "small" => 1024)
  output_channel = width_mult > 1.0 ? _make_divisible(output_channel[mode] * width_mult, 8) : output_channel[mode]
  classifier = Chain(
    Dense(explanes, output_channel, h_swish),
    Dropout(0.2),
    Dense(output_channel, nclasses),
  )

  return Chain(Chain(layers...),
               conv_bn((1, 1), inplanes, explanes, h_swish, bias = false),
               Chain(AdaptiveMeanPool((1, 1)), flatten, classifier))
end

# Model definition for MobileNetv2

struct MobileNetv2
  layers
end

"""
    MobileNetv2(width_mult::Float64 = 1.0; pretrain = false, nclasses = 1000)

Create a MobileNetv2 model with the specified configuration.
([reference](https://arxiv.org/abs/1801.04381)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet. 

# Arguments
- `width_mult`: Controls the number of channels in each layer, with 1.0 being the original
  model as detailed in the paper.
- `pretrain`: whether to load the pre-trained weights for ImageNet
- `nclasses`: the number of output classes
"""
function MobileNetv2(width_mult::Float64 = 1.0; pretrain = false, nclasses = 1000)
  layers = mobilenetv2(width_mult; nclasses)
  pretrain && loadpretrain!(model, string("MobileNetv2", config))

  MobileNetv2(layers)
end

@functor MobileNetv2

(m::MobileNetv2)(x) = m.layers(x)

backbone(m::MobileNetv2) = m.layers[1]
classifier(m::MobileNetv2) = m.layers[2:end]

# Configurations for small and large mode for MobileNetv3
mobilenetv3_configs = Dict(
  "small" => [
    # k, t, c, SE, HS, s 
    [3, 1, 16, 1, 0, 2],
    [3, 4.5, 24, 0, 0, 2],
    [3, 3.67, 24, 0, 0, 1],
    [5, 4, 40, 1, 1, 2],
    [5, 6, 40, 1, 1, 1],
    [5, 6, 40, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 6, 96, 1, 1, 2],
    [5, 6, 96, 1, 1, 1],
    [5, 6, 96, 1, 1, 1],
  ], "large" => [
    # k, t, c, SE, HS, s 
    [3, 1, 16, 0, 0, 1],
    [3, 4, 24, 0, 0, 2],
    [3, 3, 24, 0, 0, 1],
    [5, 3, 40, 1, 0, 2],
    [5, 3, 40, 1, 0, 1],
    [5, 3, 40, 1, 0, 1],
    [3, 6, 80, 0, 1, 2],
    [3, 2.5, 80, 0, 1, 1],
    [3, 2.3, 80, 0, 1, 1],
    [3, 2.3, 80, 0, 1, 1],
    [3, 6, 112, 1, 1, 1],
    [3, 6, 112, 1, 1, 1],
    [5, 6, 160, 1, 1, 2],
    [5, 6, 160, 1, 1, 1],
    [5, 6, 160, 1, 1, 1]
  ]
)

# Model definition for MobileNetv3

struct MobileNetv3
  layers
end

"""
    MobileNetv3(width_mult::Float64 = 1.0; mode = "small"; pretrain = false, nclasses = 1000)

Create a MobileNetv3 model with the specified configuration.
([reference](https://arxiv.org/abs/1905.02244)).
Set `pretrain = true` to load the model with pre-trained weights for ImageNet.

# Arguments
- `width_mult`: Controls the number of channels in each layer, with 1.0 being the original
  model as detailed in the paper.
- `mode`: 'small' or 'large' for the size of the model
- `pretrain`: whether to load the pre-trained weights for ImageNet
- `nclasses`: the number of output classes
"""
function MobileNetv3(width_mult::Float64 = 1.0; mode = "small", pretrain = false, nclasses = 1000)
  @assert mode in ["large", "small"] "`mode` has to be either \"large\" or \"small\""

  layers = mobilenetv3(mobilenetv3_configs[mode], mode, width_mult; nclasses)
  pretrain && loadpretrain!(model, string("MobileNetv3", config))
  MobileNetv3(layers)
end

@functor MobileNetv3

(m::MobileNetv3)(x) = m.layers(x)

backbone(m::MobileNetv3) = m.layers[1]
classifier(m::MobileNetv3) = m.layers[2:end]