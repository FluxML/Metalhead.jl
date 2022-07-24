"""
    _inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_3x3, pool_proj)

Create an inception module for use in GoogLeNet
([reference](https://arxiv.org/abs/1409.4842v1)).

# Arguments

  - `inplanes`: the number of input feature maps
  - `out_1x1`: the number of output feature maps for the 1x1 convolution (branch 1)
  - `red_3x3`: the number of output feature maps for the 3x3 reduction convolution (branch 2)
  - `out_3x3`: the number of output feature maps for the 3x3 convolution (branch 2)
  - `red_5x5`: the number of output feature maps for the 5x5 reduction convolution (branch 3)
  - `out_5x5`: the number of output feature maps for the 5x5 convolution (branch 3)
  - `pool_proj`: the number of output feature maps for the pooling projection (branch 4)
"""
function _inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj)
    branch1 = Chain(Conv((1, 1), inplanes => out_1x1))
    branch2 = Chain(Conv((1, 1), inplanes => red_3x3),
                    Conv((3, 3), red_3x3 => out_3x3; pad = 1))
    branch3 = Chain(Conv((1, 1), inplanes => red_5x5),
                    Conv((5, 5), red_5x5 => out_5x5; pad = 2))
    branch4 = Chain(MaxPool((3, 3); stride = 1, pad = 1),
                    Conv((1, 1), inplanes => pool_proj))
    return Parallel(cat_channels,
                    branch1, branch2, branch3, branch4)
end

"""
    googlenet(; nclasses = 1000)

Create an Inception-v1 model (commonly referred to as GoogLeNet)
([reference](https://arxiv.org/abs/1409.4842v1)).

# Arguments

  - `nclasses`: the number of output classes
"""
function googlenet(; nclasses = 1000)
    layers = Chain(Chain(Conv((7, 7), 3 => 64; stride = 2, pad = 3),
                         MaxPool((3, 3); stride = 2, pad = 1),
                         Conv((1, 1), 64 => 64),
                         Conv((3, 3), 64 => 192; pad = 1),
                         MaxPool((3, 3); stride = 2, pad = 1),
                         _inceptionblock(192, 64, 96, 128, 16, 32, 32),
                         _inceptionblock(256, 128, 128, 192, 32, 96, 64),
                         MaxPool((3, 3); stride = 2, pad = 1),
                         _inceptionblock(480, 192, 96, 208, 16, 48, 64),
                         _inceptionblock(512, 160, 112, 224, 24, 64, 64),
                         _inceptionblock(512, 128, 128, 256, 24, 64, 64),
                         _inceptionblock(512, 112, 144, 288, 32, 64, 64),
                         _inceptionblock(528, 256, 160, 320, 32, 128, 128),
                         MaxPool((3, 3); stride = 2, pad = 1),
                         _inceptionblock(832, 256, 160, 320, 32, 128, 128),
                         _inceptionblock(832, 384, 192, 384, 48, 128, 128)),
                   Chain(AdaptiveMeanPool((1, 1)),
                         MLUtils.flatten,
                         Dropout(0.4),
                         Dense(1024, nclasses)))
    return layers
end

"""
    GoogLeNet(; pretrain = false,  nclasses = 1000)

Create an Inception-v1 model (commonly referred to as `GoogLeNet`)
([reference](https://arxiv.org/abs/1409.4842v1)).

# Arguments

  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `nclasses`: the number of output classes

!!! warning
    
    `GoogLeNet` does not currently support pretrained weights.

See also [`googlenet`](#).
"""
struct GoogLeNet
    layers::Any
end
@functor GoogLeNet

function GoogLeNet(; pretrain = false, nclasses = 1000)
    layers = googlenet(; nclasses = nclasses)
    if pretrain
        loadpretrain!(layers, "GoogLeNet")
    end
    return GoogLeNet(layers)
end

(m::GoogLeNet)(x) = m.layers(x)

backbone(m::GoogLeNet) = m.layers[1]
classifier(m::GoogLeNet) = m.layers[2]
