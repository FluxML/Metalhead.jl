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
  Chain(conv_bn((3, 3), inplanes, outplanes[1]; stride = 2, pad = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2], identity; stride = 1, pad = 1, bias = false)...) :
  Chain(conv_bn((3, 3), inplanes, outplanes[1]; stride = 1, pad = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2], identity; stride = 1, pad = 1, bias = false)...)

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
  Chain(conv_bn((1, 1), inplanes, outplanes[1]; stride = 2, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 1, pad = 1, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...) :
  Chain(conv_bn((1, 1), inplanes, outplanes[1]; stride = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 1, pad = 1, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...)

"""
    skip_projection(inplanes, outplanes, downsample = false)

Create a skip projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: set to `true` to downsample the input
"""
skip_projection(inplanes, outplanes, downsample = false) = downsample ? 
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 2, bias = false)...) :
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 1, bias = false)...)

# array -> PaddedView(0, array, outplanes) for zero padding arrays
"""
    skip_identity(inplanes, outplanes[, downsample])

Create a identity projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: this argument is ignored but it is needed for compatibility with [`resnet`](#).
"""
function skip_identity(inplanes, outplanes)
  if outplanes > inplanes
    return Chain(MaxPool((1, 1), stride = 2),
                 y -> cat(y, zeros(eltype(y),
                                   size(y, 1),
                                   size(y, 2),
                                   outplanes - inplanes, size(y, 4)); dims = 3))
  else
    return identity
  end
end
skip_identity(inplanes, outplanes, downsample) = skip_identity(inplanes, outplanes)

"""
    resnet(block, residuals::NTuple{2, Any}, connection = (x, y) -> @. relu(x) + relu(y);
           channel_config, block_config, nclasses = 1000)

Create a ResNet model
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments
- `block`: a function with input `(inplanes, outplanes, downsample=false)` that returns
           a new residual block (see [`Metalhead.basicblock`](#) and [`Metalhead.bottleneck`](#))
- `residuals`: a 2-tuple of functions with input `(inplanes, outplanes, downsample=false)`,
               each of which will return a function that will be used as a new "skip" path to match a residual block.
              [`Metalhead.skip_identity`](#) and [`Metalhead.skip_projection`](#) can be used here. 
- `connection`: the binary function applied to the output of residual and skip paths in a block
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `nclasses`: the number of output classes
"""
function resnet(block, residuals::NTuple{2, Any}, connection = (x, y) -> @. relu(x) + relu(y);
                channel_config, block_config, nclasses = 1000)
  inplanes = 64
  baseplanes = 64
  layers = []
  append!(layers, conv_bn((7, 7), 3, inplanes; stride = 2, pad = (3, 3)))
  push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))
  for (i, nrepeats) in enumerate(block_config)
    # output planes within a block
    outplanes = baseplanes .* channel_config
    # push first skip connection on using first residual
    # downsample the residual path if this is the first repetition of a block
    push!(layers, Parallel(connection, block(inplanes, outplanes, i != 1),
                                       residuals[1](inplanes, outplanes[end], i != 1)))
    # push remaining skip connections on using second residual
    inplanes = outplanes[end]
    for _ in 2:nrepeats
      push!(layers, Parallel(connection, block(inplanes, outplanes, false),
                                         residuals[2](inplanes, outplanes[end], false)))
      inplanes = outplanes[end]
    end
    # next set of output plane base is doubled
    baseplanes *= 2
  end

  return Chain(Chain(layers...),
               Chain(AdaptiveMeanPool((1, 1)), flatten, Dense(inplanes, nclasses)))
end

"""
    resnet(block, shortcut_config::Symbol, connection = (x, y) -> @. relu(x) + relu(y);
           channel_config, block_config, nclasses = 1000)

Create a ResNet model
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments
- `block`: a function with input `(inplanes, outplanes, downsample=false)` that returns
           a new residual block (see [`Metalhead.basicblock`](#) and [`Metalhead.bottleneck`](#))
- `shortcut_config`: the type of shortcut style (either `:A`, `:B`, or `:C`)
    - `:A`: uses a [`Metalhead.skip_identity`](#) for all residual blocks
    - `:B`: uses a [`Metalhead.skip_projection`](#) for the first residual block
            and [`Metalhead.skip_identity`](@) for the remaining residual blocks
    - `:C`: uses a [`Metalhead.skip_projection`](#) for all residual blocks
- `connection`: the binary function applied to the output of residual and skip paths in a block
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `nclasses`: the number of output classes
"""
resnet(block, shortcut_config::Symbol, args...; kwargs...) =
  (shortcut_config == :A) ? resnet(block, (skip_identity, skip_identity), args...; kwargs...) :
  (shortcut_config == :B) ? resnet(block, (skip_projection, skip_identity), args...; kwargs...) :
  (shortcut_config == :C) ? resnet(block, (skip_projection, skip_projection), args...; kwargs...) :
  error("Unrecognized shortcut config == $shortcut_config passed to resnet (use :A, :B, or :C).")

const resnet_config =
  Dict(:resnet18 => ([1, 1], [2, 2, 2, 2], :A),
       :resnet34 => ([1, 1], [3, 4, 6, 3], :A),
       :resnet50 => ([1, 1, 4], [3, 4, 6, 3], :B),
       :resnet101 => ([1, 1, 4], [3, 4, 23, 3], :B),
       :resnet152 => ([1, 1, 4], [3, 8, 36, 3], :B))

"""
    ResNet(channel_config, block_config, shortcut_config; block, nclasses = 1000)

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
struct ResNet
  layers
end

function ResNet(channel_config, block_config, shortcut_config; block, nclasses = 1000)
  layers = resnet(block,
                  shortcut_config;
                  channel_config = channel_config,
                  block_config = block_config,
                  nclasses = nclasses)

  ResNet(layers)
end

@functor ResNet

(m::ResNet)(x) = m.layers(x)

backbone(m::ResNet) = m.layers[1]
classifier(m::ResNet) = m.layers[2]

"""
    ResNet18(; pretrain = false, nclasses = 1000)
   
Create a ResNet-18 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `nclasses`: the number of output classes

!!! warning
    `ResNet18` does not currently support pretrained weights.
"""
function ResNet18(; pretrain = false, nclasses = 1000)
  model = ResNet(resnet_config[:resnet18]...; block = basicblock, nclasses = nclasses)

  pretrain && loadpretrain!(model, "ResNet18")
  return model
end

"""
    ResNet34(; pretrain = false, nclasses = 1000)
   
Create a ResNet-34 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet34` does not currently support pretrained weights.
"""
function ResNet34(; pretrain = false, nclasses = 1000)
  model = ResNet(resnet_config[:resnet34]...; block = basicblock, nclasses = nclasses)

  pretrain && loadpretrain!(model, "ResNet34")
  return model
end

"""
    ResNet50(; pretrain = false, nclasses = 1000)
   
Create a ResNet-50 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet50` does not currently support pretrained weights.
"""
function ResNet50(; pretrain = false, nclasses = 1000)
  model = ResNet(resnet_config[:resnet50]...; block = bottleneck, nclasses = nclasses)

  pretrain && loadpretrain!(model, "ResNet50")
  return model
end

"""
    ResNet101(; pretrain = false, nclasses = 1000)
   
Create a ResNet-101 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet101` does not currently support pretrained weights.
"""
function ResNet101(; pretrain = false, nclasses = 1000)
  model = ResNet(resnet_config[:resnet101]...; block = bottleneck, nclasses = nclasses)

  pretrain && loadpretrain!(model, "ResNet101")
  return model
end

"""
    ResNet152(; pretrain = false, nclasses = 1000)
   
Create a ResNet-152 model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.ResNet`](#).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNet152` does not currently support pretrained weights.
"""
function ResNet152(; pretrain = false, nclasses = 1000)
  model = ResNet(resnet_config[:resnet152]...; block = bottleneck, nclasses = nclasses)

  pretrain && loadpretrain!(model, "ResNet152")
  return model
end
