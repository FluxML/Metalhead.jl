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
    Chain(conv_bn((3, 3), inplanes, outplanes[1]; stride = stride, pad = 1,
                  bias = false)...,
          conv_bn((3, 3), outplanes[1], outplanes[2], identity; stride = 1, pad = 1,
                  bias = false)...)
end

"""
    bottleneck(inplanes, outplanes, downsample = false; stride = [1, (downsample ? 2 : 1), 1])

Create a bottleneck residual block
([reference](https://arxiv.org/abs/1512.03385v1)). The bottleneck is composed of
3 convolutional layers each with the given `stride`.
By default, `stride` implements ["ResNet v1.5"](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch)
which uses `stride == [1, 2, 1]` when `downsample == true`.
This version is standard across various ML frameworks.
The original paper uses `stride == [2, 1, 1]` when `downsample == true` instead.

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps for each convolution
               within the residual block
- `downsample`: set to `true` to downsample the input
- `stride`: a list of the stride of the 3 convolutional layers
"""
function bottleneck(inplanes, outplanes, downsample = false;
                    stride = [1, (downsample ? 2 : 1), 1])
    Chain(conv_bn((1, 1), inplanes, outplanes[1]; stride = stride[1], bias = false)...,
          conv_bn((3, 3), outplanes[1], outplanes[2]; stride = stride[2], pad = 1,
                  bias = false)...,
          conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = stride[3],
                  bias = false)...)
end

"""
    bottleneck_v1(inplanes, outplanes, downsample = false)

Create a bottleneck residual block
([reference](https://arxiv.org/abs/1512.03385v1)). The bottleneck is composed of
3 convolutional layers with all a stride of 1 except the first convolutional
layer which has a stride of 2.

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps for each convolution
               within the residual block
- `downsample`: set to `true` to downsample the input
"""
function bottleneck_v1(inplanes, outplanes, downsample = false)
    bottleneck(inplanes, outplanes, downsample; stride = [(downsample ? 2 : 1), 1, 1])
end

"""
    resnet(block, residuals::NTuple{2, Any}, connection = addrelu;
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
function resnet(block, residuals::AbstractVector{<:NTuple{2, Any}}, connection = addrelu;
                channel_config, block_config, nclasses = 1000)
    inplanes = 64
    baseplanes = 64
    layers = []
    append!(layers, conv_bn((7, 7), 3, inplanes; stride = 2, pad = 3, bias = false))
    push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))
    for (i, nrepeats) in enumerate(block_config)
        # output planes within a block
        outplanes = baseplanes .* channel_config
        # push first skip connection on using first residual
        # downsample the residual path if this is the first repetition of a block
        push!(layers,
              Parallel(connection, block(inplanes, outplanes, i != 1),
                       residuals[i][1](inplanes, outplanes[end], i != 1)))
        # push remaining skip connections on using second residual
        inplanes = outplanes[end]
        for _ in 2:nrepeats
            push!(layers,
                  Parallel(connection, block(inplanes, outplanes, false),
                           residuals[i][2](inplanes, outplanes[end], false)))
            inplanes = outplanes[end]
        end
        # next set of output plane base is doubled
        baseplanes *= 2
    end

    return Chain(Chain(layers),
                 Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten,
                       Dense(inplanes, nclasses)))
end

"""
    resnet(block, shortcut_config::Symbol, connection = addrelu;
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
function resnet(block, shortcut_config::AbstractVector{<:Symbol}, args...; kwargs...)
    shortcut_dict = Dict(:A => (skip_identity, skip_identity),
                         :B => (skip_projection, skip_identity),
                         :C => (skip_projection, skip_projection))

    if any(sc -> !haskey(shortcut_dict, sc), shortcut_config)
        error("Unrecognized shortcut_config ($shortcut_config) passed to `resnet` (use only :A, :B, or :C).")
    end

    shortcut = [shortcut_dict[sc] for sc in shortcut_config]
    resnet(block, shortcut, args...; kwargs...)
end

function resnet(block, shortcut_config::Symbol, args...; block_config, kwargs...)
    resnet(block, fill(shortcut_config, length(block_config)), args...;
           block_config = block_config, kwargs...)
end

function resnet(block, residuals::NTuple{2}, args...; kwargs...)
    resnet(block, [residuals], args...; kwargs...)
end

const resnet_config = Dict(18 => (([1, 1], [2, 2, 2, 2], [:A, :B, :B, :B]), basicblock),
                           34 => (([1, 1], [3, 4, 6, 3], [:A, :B, :B, :B]), basicblock),
                           50 => (([1, 1, 4], [3, 4, 6, 3], [:B, :B, :B, :B]), bottleneck),
                           101 => (([1, 1, 4], [3, 4, 23, 3], [:B, :B, :B, :B]), bottleneck),
                           152 => (([1, 1, 4], [3, 8, 36, 3], [:B, :B, :B, :B]), bottleneck))

"""
    ResNet(channel_config, block_config, shortcut_config;
           block, connection = addrelu, nclasses = 1000)

Create a `ResNet` model
([reference](https://arxiv.org/abs/1512.03385v1)).
See also [`resnet`](#).

# Arguments
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `shortcut_config`: the type of shortcut style (either `:A`, `:B`, or `:C`).
   `shortcut_config` can also be a vector of symbols if different shortcut styles are applied to
   different residual blocks.
- `block`: a function with input `(inplanes, outplanes, downsample=false)` that returns
           a new residual block (see [`Metalhead.basicblock`](#) and [`Metalhead.bottleneck`](#))
- `connection`: the binary function applied to the output of residual and skip paths in a block
- `nclasses`: the number of output classes
"""
struct ResNet
    layers::Any
end

function ResNet(channel_config, block_config, shortcut_config;
                block, connection = addrelu, nclasses = 1000)
    layers = resnet(block,
                    shortcut_config,
                    connection;
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
([reference](https://arxiv.org/abs/1512.03385v1))
following [these modification](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch)
referred as ResNet v1.5.

See also [`Metalhead.resnet`](#).

# Arguments
- `depth`: depth of the ResNet model. Options include (18, 34, 50, 101, 152).
- `nclasses`: the number of output classes

!!! warning
    Only `ResNet(50)` currently supports pretrained weights.

For `ResNet(18)` and `ResNet(34)`, the parameter-free shortcut style (type `:A`)
is used in the first block and the three other blocks use type `:B` connection
(following the implementation in PyTorch). The published version of
`ResNet(18)` and `ResNet(34)` used type `:A` shortcuts for all four blocks. The
example below shows how to create a 18 or 34-layer `ResNet` using only type `:A`
shortcuts:

```julia
using Metalhead

resnet18 = ResNet([1, 1], [2, 2, 2, 2], :A; block = Metalhead.basicblock)

resnet34 = ResNet([1, 1], [3, 4, 6, 3], :A; block = Metalhead.basicblock)
```

The bottleneck of the orginal ResNet model has a stride of 2 on the first
convolutional layer when downsampling (instead of the second convolutional layers
as in ResNet v1.5). The architecture of the orignal ResNet model can be obtained
as shown below:

```julia
resnet50_v1 = ResNet([1, 1, 4], [3, 4, 6, 3], :B; block = Metalhead.bottleneck_v1)
```
"""
function ResNet(depth::Integer = 50; pretrain = false, nclasses = 1000)
    @assert depth in keys(resnet_config) "`depth` must be one of $(sort(collect(keys(resnet_config))))"

    config, block = resnet_config[depth]
    model = ResNet(config...; block = block, nclasses = nclasses)
    pretrain && loadpretrain!(model, string("ResNet", depth))
    model
end

# Compat with Metalhead 0.6; remove in 0.7
@deprecate ResNet18(; kw...) ResNet(18; kw...)
@deprecate ResNet34(; kw...) ResNet(34; kw...)
@deprecate ResNet50(; kw...) ResNet(50; kw...)
@deprecate ResNet101(; kw...) ResNet(101; kw...)
@deprecate ResNet152(; kw...) ResNet(152; kw...)
