## Inceptionv3

"""
    inceptionv3_a(inplanes, pool_proj)

Create an Inception-v3 style-A module
(ref: Fig. 5 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
  - `pool_proj`: the number of output feature maps for the pooling projection
"""
function inceptionv3_a(inplanes, pool_proj)
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
    inceptionv3_b(inplanes)

Create an Inception-v3 style-B module
(ref: Fig. 10 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inceptionv3_b(inplanes)
    branch3x3_1 = Chain(conv_bn((3, 3), inplanes, 384; stride = 2))
    branch3x3_2 = Chain(conv_bn((1, 1), inplanes, 64)...,
                        conv_bn((3, 3), 64, 96; pad = 1)...,
                        conv_bn((3, 3), 96, 96; stride = 2)...)
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
    inceptionv3_d(inplanes)

Create an Inception-v3 style-D module
(ref: [pytorch](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#L322)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inceptionv3_d(inplanes)
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
    inceptionv3_e(inplanes)

Create an Inception-v3 style-E module
(ref: Fig. 7 in [paper](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `inplanes`: number of input feature maps
"""
function inceptionv3_e(inplanes)
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
    inceptionv3(; nclasses = 1000)

Create an Inception-v3 model ([reference](https://arxiv.org/abs/1512.00567v3)).

# Arguments

  - `nclasses`: the number of output classes
"""
function inceptionv3(; nclasses = 1000)
    layer = Chain(Chain(conv_bn((3, 3), 3, 32; stride = 2)...,
                        conv_bn((3, 3), 32, 32)...,
                        conv_bn((3, 3), 32, 64; pad = 1)...,
                        MaxPool((3, 3); stride = 2),
                        conv_bn((1, 1), 64, 80)...,
                        conv_bn((3, 3), 80, 192)...,
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
                        inceptionv3_e(2048)),
                  Chain(AdaptiveMeanPool((1, 1)),
                        Dropout(0.2),
                        MLUtils.flatten,
                        Dense(2048, nclasses)))
    return layer
end

"""
    Inceptionv3(; pretrain = false, nclasses = 1000)

Create an Inception-v3 model ([reference](https://arxiv.org/abs/1512.00567v3)).
See also [`inceptionv3`](#).

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `nclasses`: the number of output classes

!!! warning
    
    `Inceptionv3` does not currently support pretrained weights.
"""
struct Inceptionv3
    layers::Any
end

function Inceptionv3(; pretrain = false, nclasses = 1000)
    layers = inceptionv3(; nclasses = nclasses)
    pretrain && loadpretrain!(layers, "Inceptionv3")
    return Inceptionv3(layers)
end

@functor Inceptionv3

(m::Inceptionv3)(x) = m.layers(x)

backbone(m::Inceptionv3) = m.layers[1]
classifier(m::Inceptionv3) = m.layers[2]

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

function inceptionv4_a()
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

function inceptionv4_b()
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

function inceptionv4_c()
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

"""
    inceptionv4(; inchannels = 3, drop_rate = 0.0, nclasses = 1000)

Create an Inceptionv4 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `inchannels`: number of input channels.
  - `drop_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.
"""
function inceptionv4(; inchannels = 3, drop_rate = 0.0, nclasses = 1000)
    body = Chain(conv_bn((3, 3), inchannels, 32; stride = 2)...,
                 conv_bn((3, 3), 32, 32)...,
                 conv_bn((3, 3), 32, 64; pad = 1)...,
                 mixed_3a(),
                 mixed_4a(),
                 mixed_5a(),
                 inceptionv4_a(),
                 inceptionv4_a(),
                 inceptionv4_a(),
                 inceptionv4_a(),
                 reduction_a(),  # mixed_6a
                 inceptionv4_b(),
                 inceptionv4_b(),
                 inceptionv4_b(),
                 inceptionv4_b(),
                 inceptionv4_b(),
                 inceptionv4_b(),
                 inceptionv4_b(),
                 reduction_b(),  # mixed_7a
                 inceptionv4_c(),
                 inceptionv4_c(),
                 inceptionv4_c())
    head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(drop_rate),
                 Dense(1536, nclasses))
    return Chain(body, head)
end

"""
    Inceptionv4(; pretrain = false, inchannels = 3, drop_rate = 0.0, nclasses = 1000)

Creates an Inceptionv4 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels.
  - `drop_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.

!!! warning
    
    `Inceptionv4` does not currently support pretrained weights.
"""
struct Inceptionv4
    layers::Any
end

function Inceptionv4(; pretrain = false, inchannels = 3, drop_rate = 0.0, nclasses = 1000)
    layers = inceptionv4(; inchannels, drop_rate, nclasses)
    pretrain && loadpretrain!(layers, "Inceptionv4")
    return Inceptionv4(layers)
end

@functor Inceptionv4

(m::Inceptionv4)(x) = m.layers(x)

backbone(m::Inceptionv4) = m.layers[1]
classifier(m::Inceptionv4) = m.layers[2]

## Inception-ResNetv2

function mixed_5b()
    branch1 = Chain(conv_bn((1, 1), 192, 96)...)
    branch2 = Chain(conv_bn((1, 1), 192, 48)...,
                    conv_bn((5, 5), 48, 64; pad = 2)...)
    branch3 = Chain(conv_bn((1, 1), 192, 64)...,
                    conv_bn((3, 3), 64, 96; pad = 1)...,
                    conv_bn((3, 3), 96, 96; pad = 1)...)
    branch4 = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                    conv_bn((1, 1), 192, 64)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function block35(scale = 1.0f0)
    branch1 = Chain(conv_bn((1, 1), 320, 32)...)
    branch2 = Chain(conv_bn((1, 1), 320, 32)...,
                    conv_bn((3, 3), 32, 32; pad = 1)...)
    branch3 = Chain(conv_bn((1, 1), 320, 32)...,
                    conv_bn((3, 3), 32, 48; pad = 1)...,
                    conv_bn((3, 3), 48, 64; pad = 1)...)
    branch4 = Chain(conv_bn((1, 1), 128, 320)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2, branch3),
                                branch4, inputscale(scale; activation = relu)), +)
end

function mixed_6a()
    branch1 = Chain(conv_bn((3, 3), 320, 384; stride = 2)...)
    branch2 = Chain(conv_bn((1, 1), 320, 256)...,
                    conv_bn((3, 3), 256, 256; pad = 1)...,
                    conv_bn((3, 3), 256, 384; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function block17(scale = 1.0f0)
    branch1 = Chain(conv_bn((1, 1), 1088, 192)...)
    branch2 = Chain(conv_bn((1, 1), 1088, 128)...,
                    conv_bn((1, 7), 128, 160; pad = (0, 3))...,
                    conv_bn((7, 1), 160, 192; pad = (3, 0))...)
    branch3 = Chain(conv_bn((1, 1), 384, 1088)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2),
                                branch3, inputscale(scale; activation = relu)), +)
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

function block8(scale = 1.0f0; activation = identity)
    branch1 = Chain(conv_bn((1, 1), 2080, 192)...)
    branch2 = Chain(conv_bn((1, 1), 2080, 192)...,
                    conv_bn((1, 3), 192, 224; pad = (0, 1))...,
                    conv_bn((3, 1), 224, 256; pad = (1, 0))...)
    branch3 = Chain(conv_bn((1, 1), 448, 2080)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2),
                                branch3, inputscale(scale; activation)), +)
end

"""
    inceptionresnetv2(; inchannels = 3, drop_rate =0.0, nclasses = 1000)

Creates an InceptionResNetv2 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `inchannels`: number of input channels.
  - `drop_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.
"""
function inceptionresnetv2(; inchannels = 3, drop_rate = 0.0, nclasses = 1000)
    body = Chain(conv_bn((3, 3), inchannels, 32; stride = 2)...,
                 conv_bn((3, 3), 32, 32)...,
                 conv_bn((3, 3), 32, 64; pad = 1)...,
                 MaxPool((3, 3); stride = 2),
                 conv_bn((3, 3), 64, 80)...,
                 conv_bn((3, 3), 80, 192)...,
                 MaxPool((3, 3); stride = 2),
                 mixed_5b(),
                 [block35(0.17f0) for _ in 1:10]...,
                 mixed_6a(),
                 [block17(0.10f0) for _ in 1:20]...,
                 mixed_7a(),
                 [block8(0.20f0) for _ in 1:9]...,
                 block8(; activation = relu),
                 conv_bn((1, 1), 2080, 1536)...)
    head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(drop_rate),
                 Dense(1536, nclasses))
    return Chain(body, head)
end

"""
    InceptionResNetv2(; pretrain = false, inchannels = 3, drop_rate =0.0, nclasses = 1000)

Creates an InceptionResNetv2 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels.
  - `drop_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.

!!! warning
    
    `InceptionResNetv2` does not currently support pretrained weights.
"""
struct InceptionResNetv2
    layers::Any
end

function InceptionResNetv2(; pretrain = false, inchannels = 3, drop_rate = 0.0,
                           nclasses = 1000)
    layers = inceptionresnetv2(; inchannels, drop_rate, nclasses)
    pretrain && loadpretrain!(layers, "InceptionResNetv2")
    return InceptionResNetv2(layers)
end

@functor InceptionResNetv2

(m::InceptionResNetv2)(x) = m.layers(x)

backbone(m::InceptionResNetv2) = m.layers[1]
classifier(m::InceptionResNetv2) = m.layers[2]

## Xception

"""
    xception_block(inchannels, outchannels, nrepeats; stride = 1, start_with_relu = true, 
                        grow_at_start = true)

Create an Xception block.
([reference](https://arxiv.org/abs/1610.02357))

# Arguments

  - `inchannels`: The number of channels in the input. The default value is 3.
  - `outchannels`: number of output channels.
  - `nrepeats`: number of repeats of depthwise separable convolution layers.
  - `stride`: stride by which to downsample the input.
  - `start_with_relu`: if true, start the block with a ReLU activation.
  - `grow_at_start`: if true, increase the number of channels at the first convolution.
"""
function xception_block(inchannels, outchannels, nrepeats; stride = 1,
                        start_with_relu = true,
                        grow_at_start = true)
    if outchannels != inchannels || stride != 1
        skip = conv_bn((1, 1), inchannels, outchannels, identity; stride = stride,
                       bias = false)
    else
        skip = [identity]
    end
    layers = []
    for i in 1:nrepeats
        if grow_at_start
            inc = i == 1 ? inchannels : outchannels
            outc = outchannels
        else
            inc = inchannels
            outc = i == nrepeats ? outchannels : inchannels
        end
        push!(layers, x -> relu.(x))
        append!(layers,
                depthwise_sep_conv_bn((3, 3), inc, outc; pad = 1, bias = false,
                                      use_bn = (false, false)))
        push!(layers, BatchNorm(outc))
    end
    layers = start_with_relu ? layers : layers[2:end]
    push!(layers, MaxPool((3, 3); stride = stride, pad = 1))
    return Parallel(+, Chain(skip...), Chain(layers...))
end

"""
    xception(; inchannels = 3, drop_rate =0.0, nclasses = 1000)

Creates an Xception model.
([reference](https://arxiv.org/abs/1610.02357))

# Arguments

  - `inchannels`: number of input channels.
  - `drop_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.
"""
function xception(; inchannels = 3, drop_rate = 0.0, nclasses = 1000)
    body = Chain(conv_bn((3, 3), inchannels, 32; stride = 2, bias = false)...,
                 conv_bn((3, 3), 32, 64; bias = false)...,
                 xception_block(64, 128, 2; stride = 2, start_with_relu = false),
                 xception_block(128, 256, 2; stride = 2),
                 xception_block(256, 728, 2; stride = 2),
                 [xception_block(728, 728, 3) for _ in 1:8]...,
                 xception_block(728, 1024, 2; stride = 2, grow_at_start = false),
                 depthwise_sep_conv_bn((3, 3), 1024, 1536; pad = 1)...,
                 depthwise_sep_conv_bn((3, 3), 1536, 2048; pad = 1)...)
    head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(drop_rate),
                 Dense(2048, nclasses))
    return Chain(body, head)
end

struct Xception
    layers::Any
end

"""
    Xception(; pretrain = false, inchannels = 3, drop_rate =0.0, nclasses = 1000)

Creates an Xception model.
([reference](https://arxiv.org/abs/1610.02357))

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet.
  - `inchannels`: number of input channels.
  - `drop_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.

!!! warning
    
    `Xception` does not currently support pretrained weights.
"""
function Xception(; pretrain = false, inchannels = 3, drop_rate = 0.0, nclasses = 1000)
    layers = xception(; inchannels, drop_rate, nclasses)
    pretrain && loadpretrain!(layers, "xception")
    return Xception(layers)
end

@functor Xception

(m::Xception)(x) = m.layers(x)

backbone(m::Xception) = m.layers[1]
classifier(m::Xception) = m.layers[2]
