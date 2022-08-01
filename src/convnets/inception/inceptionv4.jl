function mixed_3a()
    return Parallel(cat_channels,
                    MaxPool((3, 3); stride = 2),
                    Chain(conv_norm((3, 3), 64, 96; stride = 2)...))
end

function mixed_4a()
    return Parallel(cat_channels,
                    Chain(conv_norm((1, 1), 160, 64)...,
                          conv_norm((3, 3), 64, 96)...),
                    Chain(conv_norm((1, 1), 160, 64)...,
                          conv_norm((1, 7), 64, 64; pad = (0, 3))...,
                          conv_norm((7, 1), 64, 64; pad = (3, 0))...,
                          conv_norm((3, 3), 64, 96)...))
end

function mixed_5a()
    return Parallel(cat_channels,
                    Chain(conv_norm((3, 3), 192, 192; stride = 2)...),
                    MaxPool((3, 3); stride = 2))
end

function inceptionv4_a()
    branch1 = Chain(conv_norm((1, 1), 384, 96)...)
    branch2 = Chain(conv_norm((1, 1), 384, 64)...,
                    conv_norm((3, 3), 64, 96; pad = 1)...)
    branch3 = Chain(conv_norm((1, 1), 384, 64)...,
                    conv_norm((3, 3), 64, 96; pad = 1)...,
                    conv_norm((3, 3), 96, 96; pad = 1)...)
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_norm((1, 1), 384, 96)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_a()
    branch1 = Chain(conv_norm((3, 3), 384, 384; stride = 2)...)
    branch2 = Chain(conv_norm((1, 1), 384, 192)...,
                    conv_norm((3, 3), 192, 224; pad = 1)...,
                    conv_norm((3, 3), 224, 256; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function inceptionv4_b()
    branch1 = Chain(conv_norm((1, 1), 1024, 384)...)
    branch2 = Chain(conv_norm((1, 1), 1024, 192)...,
                    conv_norm((1, 7), 192, 224; pad = (0, 3))...,
                    conv_norm((7, 1), 224, 256; pad = (3, 0))...)
    branch3 = Chain(conv_norm((1, 1), 1024, 192)...,
                    conv_norm((7, 1), 192, 192; pad = (0, 3))...,
                    conv_norm((1, 7), 192, 224; pad = (3, 0))...,
                    conv_norm((7, 1), 224, 224; pad = (0, 3))...,
                    conv_norm((1, 7), 224, 256; pad = (3, 0))...)
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_norm((1, 1), 1024, 128)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_b()
    branch1 = Chain(conv_norm((1, 1), 1024, 192)...,
                    conv_norm((3, 3), 192, 192; stride = 2)...)
    branch2 = Chain(conv_norm((1, 1), 1024, 256)...,
                    conv_norm((1, 7), 256, 256; pad = (0, 3))...,
                    conv_norm((7, 1), 256, 320; pad = (3, 0))...,
                    conv_norm((3, 3), 320, 320; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function inceptionv4_c()
    branch1 = Chain(conv_norm((1, 1), 1536, 256)...)
    branch2 = Chain(conv_norm((1, 1), 1536, 384)...,
                    Parallel(cat_channels,
                             Chain(conv_norm((1, 3), 384, 256; pad = (0, 1))...),
                             Chain(conv_norm((3, 1), 384, 256; pad = (1, 0))...)))
    branch3 = Chain(conv_norm((1, 1), 1536, 384)...,
                    conv_norm((3, 1), 384, 448; pad = (1, 0))...,
                    conv_norm((1, 3), 448, 512; pad = (0, 1))...,
                    Parallel(cat_channels,
                             Chain(conv_norm((1, 3), 512, 256; pad = (0, 1))...),
                             Chain(conv_norm((3, 1), 512, 256; pad = (1, 0))...)))
    branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_norm((1, 1), 1536, 256)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

"""
    inceptionv4(; inchannels::Integer = 3, dropout_rate = 0.0, nclasses::Integer = 1000)

Create an Inceptionv4 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `inchannels`: number of input channels.
  - `dropout_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.
"""
function inceptionv4(; dropout_rate = 0.0, inchannels::Integer = 3,
                     nclasses::Integer = 1000)
    body = Chain(conv_norm((3, 3), inchannels, 32; stride = 2)...,
                 conv_norm((3, 3), 32, 32)...,
                 conv_norm((3, 3), 32, 64; pad = 1)...,
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
    head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(dropout_rate),
                 Dense(1536, nclasses))
    return Chain(body, head)
end

"""
    Inceptionv4(; pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)

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
