"""
    fire(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)

Create a fire module
([reference](https://arxiv.org/abs/1602.07360v4)).

# Arguments
- `inplanes`: number of input feature maps
- `squeeze_planes`: number of intermediate feature maps
- `expand1x1_planes`: number of output feature maps for the 1x1 expansion convolution
- `expand3x3_planes`: number of output feature maps for the 3x3 expansion convolution
"""
function fire(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)
  branch_1 = Conv((1, 1), inplanes => squeeze_planes, relu)
  branch_2 = Conv((1, 1), squeeze_planes => expand1x1_planes, relu)
  branch_3 = Conv((3, 3), squeeze_planes => expand3x3_planes, pad = 1, relu)

  return Chain(branch_1,
               Parallel(cat_channels,
                        branch_2,
                        branch_3))
end

"""
    squeezenet()

Create a SqueezeNet
([reference](https://arxiv.org/abs/1602.07360v4)).
"""
function squeezenet()
  layers = Chain(Chain(Conv((3, 3), 3 => 64, relu, stride = 2),
                       MaxPool((3, 3), stride = 2),
                       fire(64, 16, 64, 64),
                       fire(128, 16, 64, 64),
                       MaxPool((3, 3), stride = 2),
                       fire(128, 32, 128, 128),
                       fire(256, 32, 128, 128),
                       MaxPool((3, 3), stride = 2),
                       fire(256, 48, 192, 192),
                       fire(384, 48, 192, 192),
                       fire(384, 64, 256, 256),
                       fire(512, 64, 256, 256),
                       Dropout(0.5),
                       Conv((1, 1), 512 => 1000, relu)),
                 AdaptiveMeanPool((1, 1)),
                 flatten)

  return layers
end

"""
    SqueezeNet(; pretrain = false)

Create a SqueezeNet
([reference](https://arxiv.org/abs/1602.07360v4)).
Set `pretrain=true` to load the model with pre-trained weights for ImageNet.

See also [`squeezenet`](#).
"""
struct SqueezeNet
  layers
end

function SqueezeNet(; pretrain = false)
  layers = squeezenet()
  # pretrain && Flux.loadparams!(layers, weights("squeezenet"))
  pretrain && pretrain_error("SqueezeNet")

  SqueezeNet(layers)
end

@functor SqueezeNet

(m::SqueezeNet)(x) = m.layers(x)
