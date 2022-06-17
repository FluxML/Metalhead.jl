## Inceptionv3

"""
    inception3_a(inplanes, pool_proj)

Create an Inception-v3 style-A module
(ref: Fig. 5 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
  - `pool_proj`: the number of output feature maps for the pooling projection
"""
function inception3_a(inplanes, pool_proj)
    branch1x1 = Chain(conv_bn((1, 1), inplanes, 64))
    branch5x5 = Chain(conv_bn((1, 1), inplanes, 48)...,
                      conv_bn((5, 5), 48, 64; pad = 2)...)
    branch3x3 = Chain(conv_bn((1, 1), inplanes, 64)...,
                      conv_bn((3, 3), 64, 96; pad = 1)...,
                      conv_bn((3, 3), 96, 96; pad = 1)...)
    branch_pool = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                        conv_bn((1, 1), inplanes, pool_proj)...)
    return Parallel(cat_channels,
                    branch1x1, branch5x5, branch3x3, branch_pool)
end

"""
    inception3_b(inplanes)

Create an Inception-v3 style-B module
(ref: Fig. 10 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inception3_b(inplanes)
    branch3x3_1 = Chain(conv_bn((3, 3), inplanes, 384; stride = 2))
    branch3x3_2 = Chain(conv_bn((1, 1), inplanes, 64)...,
                        conv_bn((3, 3), 64, 96; pad = 1)...,
                        conv_bn((3, 3), 96, 96; stride = 2)...)
    branch_pool = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels,
                    branch3x3_1, branch3x3_2, branch_pool)
end

"""
    inception3_c(inplanes, inner_planes, n = 7)

Create an Inception-v3 style-C module
(ref: Fig. 6 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
  - `inner_planes`: the number of output feature maps within each branch
  - `n`: the "grid size" (kernel size) for the convolution layers
"""
function inception3_c(inplanes, inner_planes, n = 7)
    branch1x1 = Chain(conv_bn((1, 1), inplanes, 192))
    branch7x7_1 = Chain(conv_bn((1, 1), inplanes, inner_planes)...,
                        conv_bn((1, n), inner_planes, inner_planes; pad = (0, 3))...,
                        conv_bn((n, 1), inner_planes, 192; pad = (3, 0))...)
    branch7x7_2 = Chain(conv_bn((1, 1), inplanes, inner_planes)...,
                        conv_bn((n, 1), inner_planes, inner_planes; pad = (3, 0))...,
                        conv_bn((1, n), inner_planes, inner_planes; pad = (0, 3))...,
                        conv_bn((n, 1), inner_planes, inner_planes; pad = (3, 0))...,
                        conv_bn((1, n), inner_planes, 192; pad = (0, 3))...)
    branch_pool = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                        conv_bn((1, 1), inplanes, 192)...)
    return Parallel(cat_channels,
                    branch1x1, branch7x7_1, branch7x7_2, branch_pool)
end

"""
    inception3_d(inplanes)

Create an Inception-v3 style-D module
(ref: [pytorch](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#L322)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inception3_d(inplanes)
    branch3x3 = Chain(conv_bn((1, 1), inplanes, 192)...,
                      conv_bn((3, 3), 192, 320; stride = 2)...)
    branch7x7x3 = Chain(conv_bn((1, 1), inplanes, 192)...,
                        conv_bn((1, 7), 192, 192; pad = (0, 3))...,
                        conv_bn((7, 1), 192, 192; pad = (3, 0))...,
                        conv_bn((3, 3), 192, 192; stride = 2)...)
    branch_pool = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels,
                    branch3x3, branch7x7x3, branch_pool)
end

"""
    inception3_e(inplanes)

Create an Inception-v3 style-E module
(ref: Fig. 7 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inception3_e(inplanes)
    branch1x1 = Chain(conv_bn((1, 1), inplanes, 320))
    branch3x3_1 = Chain(conv_bn((1, 1), inplanes, 384))
    branch3x3_1a = Chain(conv_bn((1, 3), 384, 384; pad = (0, 1)))
    branch3x3_1b = Chain(conv_bn((3, 1), 384, 384; pad = (1, 0)))
    branch3x3_2 = Chain(conv_bn((1, 1), inplanes, 448)...,
                        conv_bn((3, 3), 448, 384; pad = 1)...)
    branch3x3_2a = Chain(conv_bn((1, 3), 384, 384; pad = (0, 1)))
    branch3x3_2b = Chain(conv_bn((3, 1), 384, 384; pad = (1, 0)))
    branch_pool = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                        conv_bn((1, 1), inplanes, 192)...)
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
    inception3(; nclasses = 1000)

Create an Inception-v3 model ([reference](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `nclasses`: the number of output classes

!!! warning
    
    `inception3` does not currently support pretrained weights.
"""
function inception3(; nclasses = 1000)
    layer = Chain(Chain(conv_bn((3, 3), 3, 32; stride = 2)...,
                        conv_bn((3, 3), 32, 32)...,
                        conv_bn((3, 3), 32, 64; pad = 1)...,
                        MaxPool((3, 3); stride = 2),
                        conv_bn((1, 1), 64, 80)...,
                        conv_bn((3, 3), 80, 192)...,
                        MaxPool((3, 3); stride = 2),
                        inception3_a(192, 32),
                        inception3_a(256, 64),
                        inception3_a(288, 64),
                        inception3_b(288),
                        inception3_c(768, 128),
                        inception3_c(768, 160),
                        inception3_c(768, 160),
                        inception3_c(768, 192),
                        inception3_d(768),
                        inception3_e(1280),
                        inception3_e(2048)),
                  Chain(AdaptiveMeanPool((1, 1)),
                        Dropout(0.2),
                        MLUtils.flatten,
                        Dense(2048, nclasses)))
    return layer
end

"""
    Inception3(; pretrain = false, nclasses = 1000)

Create an Inception-v3 model ([reference](https://arxiv.org/abs/1512.00567v3)).
See also [`inception3`](#).

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `nclasses`: the number of output classes

!!! warning
    
    `Inception3` does not currently support pretrained weights.
"""
struct Inception3
    layers::Any
end

function Inception3(; pretrain = false, nclasses = 1000)
    layers = inception3(; nclasses = nclasses)
    pretrain && loadpretrain!(layers, "Inception3")
    return Inception3(layers)
end

@functor Inception3

(m::Inception3)(x) = m.layers(x)

backbone(m::Inception3) = m.layers[1]
classifier(m::Inception3) = m.layers[2]

## Inceptionv4

function mixed_3a()
    return Parallel(cat_channels,
                    MaxPool((3, 3); stride = 2),
                    Chain(conv_bn((3, 3), 64, 96; stride = 2)...))
end

function mixed_4a()
    return Parallel(cat_channels,
                    Chain(conv_bn((1, 1), 160, 64)...,
                          conv_bn((3, 3), 64, 96)...),
                    Chain(conv_bn((1, 1), 160, 64)...,
                          conv_bn((1, 7), 64, 64; pad = (0, 3))...,
                          conv_bn((7, 1), 64, 64; pad = (3, 0))...,
                          conv_bn((3, 3), 64, 96)...))
end

function mixed_5a()
    return Parallel(cat_channels,
                    Chain(conv_bn((3, 3), 192, 192; stride = 2)...),
                    MaxPool((3, 3); stride = 2))
end

function inception4_a()
    branch1 = Chain(conv_bn((1, 1), 384, 96)...)
    branch2 = Chain(conv_bn((1, 1), 384, 64)...,
                    conv_bn((3, 3), 64, 96; pad = 1)...)
    branch3 = Chain(conv_bn((1, 1), 384, 64)...,
                    conv_bn((3, 3), 64, 96; pad = 1)...,
                    conv_bn((3, 3), 96, 96; pad = 1)...)
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_bn((1, 1), 384, 96)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_a()
    branch1 = Chain(conv_bn((3, 3), 384, 384; stride = 2)...)
    branch2 = Chain(conv_bn((1, 1), 384, 192)...,
                    conv_bn((3, 3), 192, 224; pad = 1)...,
                    conv_bn((3, 3), 224, 256; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function inception4_b()
    branch1 = Chain(conv_bn((1, 1), 1024, 384)...)
    branch2 = Chain(conv_bn((1, 1), 1024, 192)...,
                    conv_bn((1, 7), 192, 224; pad = (0, 3))...,
                    conv_bn((7, 1), 224, 256; pad = (3, 0))...)
    branch3 = Chain(conv_bn((1, 1), 1024, 192)...,
                    conv_bn((7, 1), 192, 192; pad = (0, 3))...,
                    conv_bn((1, 7), 192, 224; pad = (3, 0))...,
                    conv_bn((7, 1), 224, 224; pad = (0, 3))...,
                    conv_bn((1, 7), 224, 256; pad = (3, 0))...)
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_bn((1, 1), 1024, 128)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_b()
    branch1 = Chain(conv_bn((1, 1), 1024, 192)...,
                    conv_bn((3, 3), 192, 192; stride = 2)...)
    branch2 = Chain(conv_bn((1, 1), 1024, 256)...,
                    conv_bn((1, 7), 256, 256; pad = (0, 3))...,
                    conv_bn((7, 1), 256, 320; pad = (3, 0))...,
                    conv_bn((3, 3), 320, 320; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function inception4_c()
    branch1 = Chain(conv_bn((1, 1), 1536, 256)...)
    branch2 = Chain(conv_bn((1, 1), 1536, 384)...,
                    Parallel(cat_channels,
                             Chain(conv_bn((1, 3), 384, 256; pad = (0, 1))...),
                             Chain(conv_bn((3, 1), 384, 256; pad = (1, 0))...)))
    branch3 = Chain(conv_bn((1, 1), 1536, 384)...,
                    conv_bn((3, 1), 384, 448; pad = (1, 0))...,
                    conv_bn((1, 3), 448, 512; pad = (0, 1))...,
                    Parallel(cat_channels,
                             Chain(conv_bn((1, 3), 512, 256; pad = (0, 1))...),
                             Chain(conv_bn((3, 1), 512, 256; pad = (1, 0))...)))
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_bn((1, 1), 1536, 256)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function inception4(; inchannels = 3, dropout = 0.0, nclasses = 1000)
    body = Chain(conv_bn((3, 3), inchannels, 32; stride = 2)...,
                 conv_bn((3, 3), 32, 32)...,
                 conv_bn((3, 3), 32, 64; pad = 1)...,
                 mixed_3a(),
                 mixed_4a(),
                 mixed_5a(),
                 inception4_a(),
                 inception4_a(),
                 inception4_a(),
                 inception4_a(),
                 reduction_a(),  # mixed_6a
                 inception4_b(),
                 inception4_b(),
                 inception4_b(),
                 inception4_b(),
                 inception4_b(),
                 inception4_b(),
                 inception4_b(),
                 reduction_b(),  # mixed_7a
                 inception4_c(),
                 inception4_c(),
                 inception4_c())
    head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(dropout), Dense(1536, nclasses))
    return Chain(body, head)
end

struct Inception4
    layers::Any
end

function Inception4(; inchannels = 3, dropout = 0.0, nclasses = 1000)
    layers = inception4(; inchannels, dropout, nclasses)
    return Inception4(layers)
end

@functor Inception4

(m::Inception4)(x) = m.layers(x)

backbone(m::Inception4) = m.layers[1]
classifier(m::Inception4) = m.layers[2]

## Inception-ResNetv2

function mixed5_b()
    branch1 = Chain(conv_bn((1, 1), 192, 96)...)
    branch2 = Chain(conv_bn((1, 1), 192, 48)...,
                    conv_bn((5, 5), 48, 64; pad = 2)...)
    branch3 = Chain(conv_bn((1, 1), 192, 64)...,
                    conv_bn((3, 3), 64, 96; pad = 1)...,
                    conv_bn((7, 1), 64, 64; pad = 1)...)
    branch4 = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                    conv_bn((1, 1), 192, 64)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function block35(scale = 1.0)
    branch1 = Chain(conv_bn((1, 1), 320, 32)...)
    branch2 = Chain(conv_bn((1, 1), 320, 32)...,
                    conv_bn((3, 3), 32, 32)...)
    branch3 = Chain(conv_bn((1, 1), 320, 32)...,
                    conv_bn((3, 3), 32, 48; pad = 1)...,
                    conv_bn((3, 3), 48, 64; pad = 1)...)
    branch4 = Chain(conv_bn((1, 1), 128, 320)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2, branch3),
                                branch4, scale(scale; activation = relu)), +)
end

function mixed_6a()
    branch1 = Chain(conv_bn((3, 3), 320, 384; stride = 2)...)
    branch2 = Chain(conv_bn((1, 1), 320, 256)...,
                    conv_bn((3, 3), 256, 256; pad = 1)...,
                    conv_bn((3, 3), 256, 384; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function block17(scale = 1.0)
    branch1 = Chain(conv_bn((1, 1), 1088, 192)...)
    branch2 = Chain(conv_bn((1, 1), 1088, 128)...,
                    conv_bn((1, 7), 128, 160; pad = (0, 3)...),
                    conv_bn((7, 1), 160, 192; pad = (3, 0))...)
    branch3 = Chain(conv_bn((1, 1), 384, 1088)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2),
                                branch3, scale(scale; activation = relu)), +)
end

function mixed_7a()
    branch1 = Chain(conv_bn((1, 1), 1088, 256)...,
                    conv_bn((3, 3), 256, 384; stride = 2)...)
    branch2 = Chain(conv_bn((1, 1), 1088, 256)...,
                    conv_bn((3, 3), 256, 288; stride = 2)...)
    branch3 = Chain(conv_bn((1, 1), 1088, 256)...,
                    conv_bn((3, 3), 256, 288; pad = 1)...,
                    conv_bn((3, 3), 288, 320; stride = 2)...)
    branch4 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function block8(scale = 1.0; no_relu = false)
    branch1 = Chain(conv_bn((1, 1), 2080, 192)...)
    branch2 = Chain(conv_bn((1, 1), 2080, 192)...,
                    conv_bn((1, 3), 192, 224; pad = (0, 1))...,
                    conv_bn((3, 1), 224, 256; pad = (1, 0))...)
    branch3 = Chain(conv_bn((1, 1), 448, 2080)...)
    activation = no_relu ? identity : relu
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2),
                                branch3, scale(scale; activation = activation)), +)
end

function inception_resnetv2(; inchannels = 3, dropout = 0.0, nclasses = 1000)
    body = Chain(conv_bn((3, 3), inchannels, 32; stride = 2)...,
                 conv_bn((3, 3), 32, 32)...,
                 conv_bn((3, 3), 32, 64; pad = 1)...,
                 MaxPool((3, 3); stride = 2),
                 conv_bn((3, 3), 64, 80)...,
                 conv_bn((3, 3), 80, 192)...,
                 MaxPool((3, 3); stride = 2),
                 mixed_5b(),
                 [block35(0.17) for _ in 1:10]...,
                 mixed_6a(),
                 [block17(0.10) for _ in 1:20]...,
                 mixed_7a(),
                 [block8(0.20) for _ in 1:9]...,
                 block8(; no_relu = true),
                 conv_bn((1, 1), 2080, 1536)...)
    head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(dropout), Dense(1536, nclasses))
    return Chain(body, head)
end

struct Inception_ResNetv2
    layers::Any
end

function Inception_ResNetv2(; inchannels = 3, dropout = 0.0, nclasses = 1000)
    layers = inception_resnetv2(; inchannels, dropout, nclasses)
    return Inception_ResNetv2(layers)
end

@functor Inception_ResNetv2

(m::Inception_ResNetv2)(x) = m.layers(x)

backbone(m::Inception_ResNetv2) = m.layers[1]
classifier(m::Inception_ResNetv2) = m.layers[2]
