"""
    inceptionv3_a(inplanes, pool_proj)

Create an Inception-v3 style-A module
(ref: Fig. 5 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
  - `pool_proj`: the number of output feature maps for the pooling projection
"""
function inceptionv3_a(inplanes, pool_proj)
    branch1x1 = Chain(conv_norm((1, 1), inplanes, 64))
    branch5x5 = Chain(conv_norm((1, 1), inplanes, 48)...,
                      conv_norm((5, 5), 48, 64; pad = 2)...)
    branch3x3 = Chain(conv_norm((1, 1), inplanes, 64)...,
                      conv_norm((3, 3), 64, 96; pad = 1)...,
                      conv_norm((3, 3), 96, 96; pad = 1)...)
    branch_pool = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                        conv_norm((1, 1), inplanes, pool_proj)...)
    return Parallel(cat_channels,
                    branch1x1, branch5x5, branch3x3, branch_pool)
end

"""
    inceptionv3_b(inplanes)

Create an Inception-v3 style-B module
(ref: Fig. 10 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inceptionv3_b(inplanes)
    branch3x3_1 = Chain(conv_norm((3, 3), inplanes, 384; stride = 2))
    branch3x3_2 = Chain(conv_norm((1, 1), inplanes, 64)...,
                        conv_norm((3, 3), 64, 96; pad = 1)...,
                        conv_norm((3, 3), 96, 96; stride = 2)...)
    branch_pool = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels,
                    branch3x3_1, branch3x3_2, branch_pool)
end

"""
    inceptionv3_c(inplanes, inner_planes, n = 7)

Create an Inception-v3 style-C module
(ref: Fig. 6 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
  - `inner_planes`: the number of output feature maps within each branch
  - `n`: the "grid size" (kernel size) for the convolution layers
"""
function inceptionv3_c(inplanes, inner_planes, n = 7)
    branch1x1 = Chain(conv_norm((1, 1), inplanes, 192))
    branch7x7_1 = Chain(conv_norm((1, 1), inplanes, inner_planes)...,
                        conv_norm((1, n), inner_planes, inner_planes; pad = (0, 3))...,
                        conv_norm((n, 1), inner_planes, 192; pad = (3, 0))...)
    branch7x7_2 = Chain(conv_norm((1, 1), inplanes, inner_planes)...,
                        conv_norm((n, 1), inner_planes, inner_planes; pad = (3, 0))...,
                        conv_norm((1, n), inner_planes, inner_planes; pad = (0, 3))...,
                        conv_norm((n, 1), inner_planes, inner_planes; pad = (3, 0))...,
                        conv_norm((1, n), inner_planes, 192; pad = (0, 3))...)
    branch_pool = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                        conv_norm((1, 1), inplanes, 192)...)
    return Parallel(cat_channels,
                    branch1x1, branch7x7_1, branch7x7_2, branch_pool)
end

"""
    inceptionv3_d(inplanes)

Create an Inception-v3 style-D module
(ref: [pytorch](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#L322)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inceptionv3_d(inplanes)
    branch3x3 = Chain(conv_norm((1, 1), inplanes, 192)...,
                      conv_norm((3, 3), 192, 320; stride = 2)...)
    branch7x7x3 = Chain(conv_norm((1, 1), inplanes, 192)...,
                        conv_norm((1, 7), 192, 192; pad = (0, 3))...,
                        conv_norm((7, 1), 192, 192; pad = (3, 0))...,
                        conv_norm((3, 3), 192, 192; stride = 2)...)
    branch_pool = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels,
                    branch3x3, branch7x7x3, branch_pool)
end

"""
    inceptionv3_e(inplanes)

Create an Inception-v3 style-E module
(ref: Fig. 7 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inceptionv3_e(inplanes)
    branch1x1 = Chain(conv_norm((1, 1), inplanes, 320))
    branch3x3_1 = Chain(conv_norm((1, 1), inplanes, 384))
    branch3x3_1a = Chain(conv_norm((1, 3), 384, 384; pad = (0, 1)))
    branch3x3_1b = Chain(conv_norm((3, 1), 384, 384; pad = (1, 0)))
    branch3x3_2 = Chain(conv_norm((1, 1), inplanes, 448)...,
                        conv_norm((3, 3), 448, 384; pad = 1)...)
    branch3x3_2a = Chain(conv_norm((1, 3), 384, 384; pad = (0, 1)))
    branch3x3_2b = Chain(conv_norm((3, 1), 384, 384; pad = (1, 0)))
    branch_pool = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                        conv_norm((1, 1), inplanes, 192)...)
    return Parallel(cat_channels,
                    branch1x1,
                    Chain(branch3x3_1,
                          Parallel(cat_channels,
                                   branch3x3_1a, branch3x3_1b)),
                    Chain(branch3x3_2,
                          Parallel(cat_channels,
                                   branch3x3_2a, branch3x3_2b)),
                    branch_pool)
end

"""
    inceptionv3(; inchannels::Integer = 3, nclasses::Integer = 1000)

Create an Inception-v3 model ([reference](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `nclasses`: the number of output classes
"""
function inceptionv3(; inchannels::Integer = 3, nclasses::Integer = 1000)
    backbone = Chain(conv_norm((3, 3), inchannels, 32; stride = 2)...,
                     conv_norm((3, 3), 32, 32)...,
                     conv_norm((3, 3), 32, 64; pad = 1)...,
                     MaxPool((3, 3); stride = 2),
                     conv_norm((1, 1), 64, 80)...,
                     conv_norm((3, 3), 80, 192)...,
                     MaxPool((3, 3); stride = 2),
                     inceptionv3_a(192, 32),
                     inceptionv3_a(256, 64),
                     inceptionv3_a(288, 64),
                     inceptionv3_b(288),
                     inceptionv3_c(768, 128),
                     inceptionv3_c(768, 160),
                     inceptionv3_c(768, 160),
                     inceptionv3_c(768, 192),
                     inceptionv3_d(768),
                     inceptionv3_e(1280),
                     inceptionv3_e(2048))
    return Chain(backbone, create_classifier(2048, nclasses; dropout_rate = 0.2))
end

"""
    Inceptionv3(; pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)

Create an Inception-v3 model ([reference](https://arxiv.org/abs/1512.00567v3)).
See also [`inceptionv3`](#).

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels
  - `nclasses`: the number of output classes

!!! warning
    
    `Inceptionv3` does not currently support pretrained weights.
"""
struct Inceptionv3
    layers::Any
end
@functor Inceptionv3

function Inceptionv3(; pretrain::Bool = false, inchannels::Integer = 3,
                     nclasses::Integer = 1000)
    layers = inceptionv3(; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "Inceptionv3")
    end
    return Inceptionv3(layers)
end

(m::Inceptionv3)(x) = m.layers(x)

backbone(m::Inceptionv3) = m.layers[1]
classifier(m::Inceptionv3) = m.layers[2]
