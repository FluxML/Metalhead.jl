"""
    fire(inplanes::Integer, squeeze_planes::Integer, expand1x1_planes::Integer,
         expand3x3_planes::Integer)

Create a fire module
([reference](https://arxiv.org/abs/1602.07360v4)).

# Arguments

  - `inplanes`: number of input feature maps
  - `squeeze_planes`: number of intermediate feature maps
  - `expand1x1_planes`: number of output feature maps for the 1x1 expansion convolution
  - `expand3x3_planes`: number of output feature maps for the 3x3 expansion convolution
"""
function fire(inplanes::Integer, squeeze_planes::Integer, expand1x1_planes::Integer,
              expand3x3_planes::Integer)
    branch_1 = Conv((1, 1), inplanes => squeeze_planes, relu)
    branch_2 = Conv((1, 1), squeeze_planes => expand1x1_planes, relu)
    branch_3 = Conv((3, 3), squeeze_planes => expand3x3_planes, relu; pad = 1)
    return Chain(branch_1, Parallel(cat_channels, branch_2, branch_3))
end

"""
    squeezenet(; inchannels::Integer = 3, nclasses::Integer = 1000)

Create a SqueezeNet
([reference](https://arxiv.org/abs/1602.07360v4)).

# Arguments

  - `inchannels`: number of input channels.
  - `nclasses`: the number of output classes.
"""
function squeezenet(; inchannels::Integer = 3, nclasses::Integer = 1000)
    backbone = Chain(Conv((3, 3), inchannels => 64, relu; stride = 2),
                     MaxPool((3, 3); stride = 2),
                     fire(64, 16, 64, 64),
                     fire(128, 16, 64, 64),
                     MaxPool((3, 3); stride = 2),
                     fire(128, 32, 128, 128),
                     fire(256, 32, 128, 128),
                     MaxPool((3, 3); stride = 2),
                     fire(256, 48, 192, 192),
                     fire(384, 48, 192, 192),
                     fire(384, 64, 256, 256),
                     fire(512, 64, 256, 256))
    classifier = Chain(Dropout(0.5), Conv((1, 1), 512 => nclasses, relu),
                       AdaptiveMeanPool((1, 1)), MLUtils.flatten)
    return Chain(backbone, classifier)
end

"""
    SqueezeNet(; pretrain::Bool = false, inchannels::Integer = 3,
           nclasses::Integer = 1000)

Create a SqueezeNet
([reference](https://arxiv.org/abs/1602.07360v4)).

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels.
  - `nclasses`: the number of output classes.

!!! warning
    
    `SqueezeNet` does not currently support pretrained weights.

See also [`squeezenet`](#).
"""
struct SqueezeNet
    layers::Any
end
@functor SqueezeNet

function SqueezeNet(; pretrain::Bool = false, inchannels::Integer = 3,
                    nclasses::Integer = 1000)
    layers = squeezenet(; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "SqueezeNet")
    end
    return SqueezeNet(layers)
end

(m::SqueezeNet)(x) = m.layers(x)

backbone(m::SqueezeNet) = m.layers[1]
classifier(m::SqueezeNet) = m.layers[2:end]
