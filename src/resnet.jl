"""
    basicblock(inplanes, outplanes, downsample = false)

Create a basic residual block
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps for each convolution
               within the residual block
- `downsample`: set to `true` to downsample the input
"""
basicblock(inplanes, outplanes, downsample = false) = downsample ?
  Chain(conv_bn((3, 3), inplanes, outplanes[1]; stride=2, pad=1, usebias=false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride=1, pad=1, usebias=false)...) :
  Chain(conv_bn((3, 3), inplanes, outplanes[1]; stride=1, pad=1, usebias=false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride=1, pad=1, usebias=false)...)

"""
    bottleneck(inplanes, outplanes, downsample = false)

Create a bottleneck residual block
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps for each convolution
               within the residual block
- `downsample`: set to `true` to downsample the input
"""
bottleneck(inplanes, outplanes, downsample = false) = downsample ?
  Chain(conv_bn((1, 1), inplanes, outplanes[1]; stride=2, usebias=false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride=1, pad=1, usebias=false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3]; stride=1, usebias=false)...) :
  Chain(conv_bn((1, 1), inplanes, outplanes[1]; stride=1, usebias=false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride=1, pad=1, usebias=false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3]; stride=1, usebias=false)...)

"""
    skip_projection(inplanes, outplanes, downsample = false)

Create a skip projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps
- `downsample`: set to `true` to downsample the input
"""
skip_projection(inplanes, outplanes, downsample = false) = downsample ? 
  Chain(conv_bn((1, 1), inplanes, outplanes; stride=2, usebias=false)...) :
  Chain(conv_bn((1, 1), inplanes, outplanes; stride=1, usebias=false)...)

# array -> PaddedView(0, array, outplanes) for zero padding arrays
"""
    skip_identity(inplanes, outplanes)

Create a identity projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps
"""
function skip_identity(inplanes, outplanes)
  if outplanes[end] > inplanes
    return Chain(MaxPool((1, 1), stride = 2),
                 y -> cat(y, zeros(eltype(y),
                                   size(y, 1),
                                   size(y, 2),
                                   outplanes[end] - inplanes, size(y, 4)); dims = 3))
  else
    return identity
  end
end

"""
    resnet(; block, shortcut_config, channel_config, block_config)

Create a ResNet model
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments
- `block`: a function with input `(inplanes, outplanes, downsample=false)` that returns
           a new residual block (see [`Metalhead.basicblock`](#) and [`Metalhead.bottleneck`](#))
- `shortcut_config`: the type of shortcut style (either `:A`, `:B`, or `:C`)
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `nclasses`: the number of output classes
"""
function resnet(; block, shortcut_config, channel_config, block_config, nclasses=1000)
  inplanes = 64
  baseplanes = 64
  layers = []
  append!(layers, conv_bn((7, 7), 3, inplanes; stride=2, pad=(3, 3)))
  push!(layers, MaxPool((3, 3), stride=(2, 2), pad=(1, 1)))
  for (i, nrepeats) in enumerate(block_config)
    outplanes = baseplanes .* channel_config
    if shortcut_config == :A
      push!(layers, Parallel(+, block(inplanes, outplanes, i != 1),
                                skip_identity(inplanes, outplanes)))
    elseif shortcut_config == :B || shortcut_config == :C
      push!(layers, Parallel(+, block(inplanes, outplanes, i != 1),
                                skip_projection(inplanes, outplanes[end], i != 1)))
    end
    inplanes = outplanes[end]
    for j in 2:nrepeats
      if shortcut_config == :A || shortcut_config == :B
        push!(layers, Parallel(+, block(inplanes, outplanes, false),
                                  skip_identity(inplanes, outplanes[end])))
      elseif shortcut_config == :C
        push!(layers, Parallel(+, block(inplanes, outplanes, false),
                                  skip_projection(inplanes, outplanes, false)))
      end
      inplanes = outplanes[end]
    end
    baseplanes *= 2
  end
  push!(layers, AdaptiveMeanPool((1, 1)))
  push!(layers, flatten)
  push!(layers, Dense(inplanes, nclasses))

  return Chain(layers...)
end

const resnet_config =
  Dict(:resnet18 => ([1, 1], [2, 2, 2, 2], :A),
       :resnet34 => ([1, 1], [3, 4, 6, 3], :A),
       :resnet50 => ([1, 1, 4], [3, 4, 6, 3], :B),
       :resnet101 => ([1, 1, 4], [3, 4, 23, 3], :B),
       :resnet152 => ([1, 1, 4], [3, 8, 36, 3], :B))

"""
    ResNet(channel_config, block_config, shortcut_config; block, nclasses=1000)

Create a `ResNet` model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`resnet`](#).

# Arguments
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `shortcut_config`: the type of shortcut style (either `:A`, `:B`, or `:C`)
- `block`: a function with input `(inplanes, outplanes, downsample=false)` that returns
           a new residual block (see [`Metalhead.basicblock`](#) and [`Metalhead.bottleneck`](#))
- `nclasses`: the number of output classes
"""
struct ResNet{T}
  layers::T
end

function ResNet(channel_config, block_config, shortcut_config; block, nclasses=1000)
  layers = resnet(block=block,
                  shortcut_config=shortcut_config,
                  channel_config=channel_config,
                  block_config=block_config,
                  nclasses=nclasses)

  ResNet(layers)
end

@funtor ResNet

(m::ResNet)(x) = m.layers(x)

"""
    ResNet18(; pretrain=false, nclasses=1000)
   
Create a ResNet-18 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `nclasses`: the number of output classes

!!! warning
    `ResNet18` does not currently support pretrained weights.
"""
function ResNet18(; pretrain=false, nclasses=1000)
  model = ResNet(resnet_config[:resnet18]...; block=basicblock, nclasses=nclasses)

  pretrain && pretrain_error("ResNet18")
  return model
end

"""
    ResNet34(; pretrain=false, nclasses=1000)
   
Create a ResNet-34 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet34` does not currently support pretrained weights.
"""
function ResNet34(; pretrain=false, nclasses=1000)
  model = ResNet(resnet_config[:resnet34]...; nclasses=nclasses)

  pretrain && pretrain_error("ResNet34")
  return model
end

"""
    ResNet50(; pretrain=false, nclasses=1000)
   
Create a ResNet-50 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes
"""
function ResNet50(; pretrain=false, nclasses=1000)
  model = ResNet(resnet_config[:resnet50]...; nclasses=nclasses)

  pretrain && Flux.loadparams!(model.layers, weights("resnet50"))
  return model
end

"""
    ResNet101(; pretrain=false, nclasses=1000)
   
Create a ResNet-101 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet101` does not currently support pretrained weights.
"""
function ResNet101(; pretrain=false, nclasses=1000)
  model = ResNet(resnet_config[:resnet101]...; nclasses=nclasses)

  pretrain && pretrain_error("ResNet101")
  return model
end

"""
    ResNet152(; pretrain=false, nclasses=1000)
   
Create a ResNet-152 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet152` does not currently support pretrained weights.
"""
function ResNet152(; pretrain=false, nclasses=1000)
  model = ResNet(resnet_config[:resnet152]...; nclasses=nclasses)

  pretrain && pretrain_error("ResNet152")
  return model
end
