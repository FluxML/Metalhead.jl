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
function basicblock(inplanes, outplanes, downsample = false)
  stride = downsample ? 2 : 1
  Chain(conv_bn((3, 3), inplanes, outplanes[1]; stride = stride, pad = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2], identity; stride = 1, pad = 1, bias = false)...)
end

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
function bottleneck(inplanes, outplanes, downsample = false)
  stride = downsample ? 2 : 1
  Chain(conv_bn((1, 1), inplanes, outplanes[1]; stride = stride, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 1, pad = 1, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...)
end

"""
    skip_projection(inplanes, outplanes, downsample = false)

Create a skip projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: set to `true` to downsample the input
"""
function skip_projection(inplanes, outplanes, downsample = false)
  stride = downsample ? 2 : 1 
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = stride, bias = false)...)
end

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
function resnet(block, shortcut_config::Symbol, args...; kwargs...)
  shortcut = if shortcut_config == :A
      (skip_identity, skip_identity)
    elseif shortcut_config == :B
      (skip_projection, skip_identity)
    elseif shortcut_config == :C
      (skip_projection, skip_projection)
    else
      error("Unrecognized shortcut_config ($shortcut_config) passed to `resnet` (use :A, :B, or :C).")
  end
  resnet(block, shortcut, args...; kwargs...)
end

const resnet_config =
  Dict(18 => (([1, 1], [2, 2, 2, 2], :A), basicblock),
       34 => (([1, 1], [3, 4, 6, 3], :A), basicblock),
       50 => (([1, 1, 4], [3, 4, 6, 3], :B), bottleneck),
       101 => (([1, 1, 4], [3, 4, 23, 3], :B), bottleneck),
       152 => (([1, 1, 4], [3, 8, 36, 3], :B), bottleneck))

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
    ResNet(depth = 50; pretrain = false, nclasses = 1000)
   
Create a ResNet model with a specified depth
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`Metalhead.resnet`](#).

# Arguments
- `depth`: depth of the ResNet model. Options include (18, 34, 50, 101, 152).
- `nclasses`: the number of output classes

!!! warning
    Only `ResNet(50)` currently supports pretrained weights.
"""
function ResNet(depth::Int = 50; pretrain = false, nclasses = 1000)
    @assert depth in keys(resnet_config) "`depth` must be one of $(sort(collect(keys(resnet_config))))"

    config, block = resnet_config[depth]
    model = ResNet(config...; block = block, nclasses = nclasses)
    pretrain && loadpretrain!(model, string("ResNet", depth))
    model
end

# Compat with Methalhead 0.6; remove in 0.7
@deprecate ResNet18(; kw...) ResNet(18; kw...)
@deprecate ResNet34(; kw...) ResNet(34; kw...)
@deprecate ResNet50(; kw...) ResNet(50; kw...)
@deprecate ResNet101(; kw...) ResNet(101; kw...)
@deprecate ResNet152(; kw...) ResNet(152; kw...)
