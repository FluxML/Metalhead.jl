function mixed_3a()
    return Parallel(cat_channels,
                    MaxPool((3, 3); stride = 2),
                    Chain(basic_conv_bn((3, 3), 64, 96; stride = 2)...))
end

function mixed_4a()
    return Parallel(cat_channels,
                    Chain(basic_conv_bn((1, 1), 160, 64)...,
                          basic_conv_bn((3, 3), 64, 96)...),
                    Chain(basic_conv_bn((1, 1), 160, 64)...,
                          basic_conv_bn((7, 1), 64, 64; pad = (3, 0))...,
                          basic_conv_bn((1, 7), 64, 64; pad = (0, 3))...,
                          basic_conv_bn((3, 3), 64, 96)...))
end

function mixed_5a()
    return Parallel(cat_channels,
                    Chain(basic_conv_bn((3, 3), 192, 192; stride = 2)...),
                    MaxPool((3, 3); stride = 2))
end

function inceptionv4_a()
    branch1 = Chain(basic_conv_bn((1, 1), 384, 96)...)
    branch2 = Chain(basic_conv_bn((1, 1), 384, 64)...,
                    basic_conv_bn((3, 3), 64, 96; pad = 1)...)
    branch3 = Chain(basic_conv_bn((1, 1), 384, 64)...,
                    basic_conv_bn((3, 3), 64, 96; pad = 1)...,
                    basic_conv_bn((3, 3), 96, 96; pad = 1)...)
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1),
                    basic_conv_bn((1, 1), 384, 96)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_a()
    branch1 = Chain(basic_conv_bn((3, 3), 384, 384; stride = 2)...)
    branch2 = Chain(basic_conv_bn((1, 1), 384, 192)...,
                    basic_conv_bn((3, 3), 192, 224; pad = 1)...,
                    basic_conv_bn((3, 3), 224, 256; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function inceptionv4_b()
    branch1 = Chain(basic_conv_bn((1, 1), 1024, 384)...)
    branch2 = Chain(basic_conv_bn((1, 1), 1024, 192)...,
                    basic_conv_bn((7, 1), 192, 224; pad = (0, 3))...,
                    basic_conv_bn((1, 7), 224, 256; pad = (3, 0))...)
    branch3 = Chain(basic_conv_bn((1, 1), 1024, 192)...,
                    basic_conv_bn((1, 7), 192, 192; pad = (3, 0))...,
                    basic_conv_bn((7, 1), 192, 224; pad = (0, 3))...,
                    basic_conv_bn((1, 7), 224, 224; pad = (3, 0))...,
                    basic_conv_bn((7, 1), 224, 256; pad = (0, 3))...)
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1),
                    basic_conv_bn((1, 1), 1024, 128)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_b()
    branch1 = Chain(basic_conv_bn((1, 1), 1024, 192)...,
                    basic_conv_bn((3, 3), 192, 192; stride = 2)...)
    branch2 = Chain(basic_conv_bn((1, 1), 1024, 256)...,
                    basic_conv_bn((7, 1), 256, 256; pad = (3, 0))...,
                    basic_conv_bn((1, 7), 256, 320; pad = (0, 3))...,
                    basic_conv_bn((3, 3), 320, 320; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function inceptionv4_c()
    branch1 = Chain(basic_conv_bn((1, 1), 1536, 256)...)
    branch2 = Chain(basic_conv_bn((1, 1), 1536, 384)...,
                    Parallel(cat_channels,
                             Chain(basic_conv_bn((3, 1), 384, 256; pad = (1, 0))...),
                             Chain(basic_conv_bn((1, 3), 384, 256; pad = (0, 1))...)))
    branch3 = Chain(basic_conv_bn((1, 1), 1536, 384)...,
                    basic_conv_bn((1, 3), 384, 448; pad = (0, 1))...,
                    basic_conv_bn((3, 1), 448, 512; pad = (1, 0))...,
                    Parallel(cat_channels,
                             Chain(basic_conv_bn((3, 1), 512, 256; pad = (1, 0))...),
                             Chain(basic_conv_bn((1, 3), 512, 256; pad = (0, 1))...)))
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1),
                    basic_conv_bn((1, 1), 1536, 256)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

"""
    inceptionv4(; inchannels::Integer = 3, dropout_prob = nothing, nclasses::Integer = 1000)

Create an Inceptionv4 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `inchannels`: number of input channels.
  - `dropout_prob`: probability of dropout in classifier head. Set to `nothing` to disable dropout.
  - `nclasses`: the number of output classes.
"""
function inceptionv4(; dropout_prob = nothing, inchannels::Integer = 3,
                     nclasses::Integer = 1000)
    backbone = Chain(basic_conv_bn((3, 3), inchannels, 32; stride = 2)...,
                     basic_conv_bn((3, 3), 32, 32)...,
                     basic_conv_bn((3, 3), 32, 64; pad = 1)...,
                     mixed_3a(), mixed_4a(), mixed_5a(),
                     [inceptionv4_a() for _ in 1:4]...,
                     reduction_a(),  # mixed_6a
                     [inceptionv4_b() for _ in 1:7]...,
                     reduction_b(),  # mixed_7a
                     [inceptionv4_c() for _ in 1:3]...)
    return Chain(backbone, create_classifier(1536, nclasses; dropout_prob))
end

"""
    Inceptionv4(; pretrain::Bool = false, inchannels::Integer = 3,
                nclasses::Integer = 1000)

Creates an Inceptionv4 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels.
  - `nclasses`: the number of output classes.

!!! warning
    
    `Inceptionv4` does not currently support pretrained weights.
"""
struct Inceptionv4
    layers::Any
end
@functor Inceptionv4

function Inceptionv4(; pretrain::Bool = false, inchannels::Integer = 3,
                     nclasses::Integer = 1000)
    layers = inceptionv4(; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "Inceptionv4")
    end
    return Inceptionv4(layers)
end

(m::Inceptionv4)(x) = m.layers(x)

backbone(m::Inceptionv4) = m.layers[1]
classifier(m::Inceptionv4) = m.layers[2]
