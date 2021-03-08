"""
    dense_bottleneck(inplanes, growth_rate)

Create a Densenet bottleneck layer
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments
- `inplanes`: number of input feature maps
- `growth_rate`: number of output feature maps
                 (and scaling factor for inner feature maps; see ref)
"""
function dense_bottleneck(inplanes, growth_rate)
  inner_channels = 4 * growth_rate
  m = Chain(conv_bn((1, 1), inplanes, inner_channels; usebias=false, rev=true)...,
            conv_bn((3, 3), inner_channels, growth_rate; pad=1, usebias=false, rev=true)...)

  SkipConnection(m, (mx, x) -> cat(x, mx; dims=3))
end

"""
    transition(inplanes, outplanes)

Create a DenseNet transition sequence
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments
- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
"""
transition(inplanes, outplanes) = (conv_bn((1, 1), inplanes, outplanes; usebias=false, rev=true)...,
                                   MeanPool((2, 2)))

"""
    dense_block(inplanes, growth_rate, nblock)

Create a sequence of `nblock` DesNet bottlenecks with `growth_rate`
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments
- `inplanes`: number of input feature maps to the full sequence
- `growth_rate`: the rate at which output feature maps grow across blocks
- `nblock`: the number of blocks
"""
function dense_block(inplanes, growth_rate, nblock)
  layers = []
  for i in 1:nblock
    push!(layers, dense_bottleneck(inplanes, growth_rate))
    inplanes += growth_rate
  end
  return layers
end

"""
    densenet(nblocks=(6, 12, 24, 16); growth_rate=32, reduction=0.5, num_classes=1000)

Create a DenseNet model
([reference](https://arxiv.org/abs/1608.06993)).

# Arguments
- `nblocks`: number of dense blocks between transitions
- `growth_rate`: the output feature map growth rate of dense blocks (i.e. `k` in the paper)
- `reduction`: the factor by which the number of feature maps is scaled across each transition
- `num_classes`: the number of output classes
"""
function densenet(nblocks=(6, 12, 24, 16); growth_rate=32, reduction=0.5, num_classes=1000)
  num_planes = 2 * growth_rate
  layers = []
  append!(layers, conv_bn((7, 7), 3, num_planes; stride=2, pad=(3, 3), usebias=false))
  push!(layers, MaxPool((3, 3), stride=2, pad=(1, 1)))

  for i in 1:3
    append!(layers, dense_block(num_planes, growth_rate, nblocks[i]))
    num_planes += nblocks[i] * growth_rate
    out_planes = Int(floor(num_planes * reduction))
    append!(layers, transition(num_planes, out_planes))
    num_planes = out_planes
  end

  append!(layers, dense_block(num_planes, growth_rate, nblocks[4]))
  num_planes += nblocks[4] * growth_rate
  push!(layers, BatchNorm(num_planes, relu))

  return Chain(layers...,
               AdaptiveMeanPool((1, 1)),
               flatten,
               Dense(num_planes, num_classes))
end

"""
    densenet121(; pretrain=false)

Create a DenseNet-121 model
([reference](https://arxiv.org/abs/1608.06993)).
Set `pretrain=true` to load the model with pre-trained weights for ImageNet.

See also [`Metalhead.densenet`](#).
"""
function densenet121(; pretrain=false)
  model = densenet()

  pretrain && Flux.loadparams!(model, weights("densenet121"))
end

"""
    densenet161(; pretrain=false)

Create a DenseNet-161 model
([reference](https://arxiv.org/abs/1608.06993)).

!!! warning
    `densenet161` does not currently support pretrained weights.

See also [`Metalhead.densenet`](#).
"""
function densenet161(; pretrain=false)
  model = densenet((6, 12, 36, 24); growth_rate=64)

  pretrain && pretrain_error("densenet161")
  return model
end

"""
    densenet169(; pretrain=false)

Create a DenseNet-169 model
([reference](https://arxiv.org/abs/1608.06993)).

!!! warning
    `densenet169` does not currently support pretrained weights.

See also [`Metalhead.densenet`](#).
"""
function densenet169(; pretrain=false)
  model = densenet((6, 12, 32, 32))

  pretrain && pretrain_error("densenet169")
  return model
end

"""
    densenet201(; pretrain=false)

Create a DenseNet-201 model
([reference](https://arxiv.org/abs/1608.06993)).

!!! warning
    `densenet201` does not currently support pretrained weights.

See also [`Metalhead.densenet`](#).
"""
function densenet201(; pretrain=false)
  model = densenet((6, 12, 48, 32))

  pretrain && pretrain_error("densenet201")
  return model
end