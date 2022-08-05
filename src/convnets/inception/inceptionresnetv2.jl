function mixed_5b()
    branch1 = Chain(conv_norm((1, 1), 192, 96)...)
    branch2 = Chain(conv_norm((1, 1), 192, 48)...,
                    conv_norm((5, 5), 48, 64; pad = 2)...)
    branch3 = Chain(conv_norm((1, 1), 192, 64)...,
                    conv_norm((3, 3), 64, 96; pad = 1)...,
                    conv_norm((3, 3), 96, 96; pad = 1)...)
    branch4 = Chain(MeanPool((3, 3); pad = 1, stride = 1),
                    conv_norm((1, 1), 192, 64)...)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function block35(scale = 1.0f0)
    branch1 = Chain(conv_norm((1, 1), 320, 32)...)
    branch2 = Chain(conv_norm((1, 1), 320, 32)...,
                    conv_norm((3, 3), 32, 32; pad = 1)...)
    branch3 = Chain(conv_norm((1, 1), 320, 32)...,
                    conv_norm((3, 3), 32, 48; pad = 1)...,
                    conv_norm((3, 3), 48, 64; pad = 1)...)
    branch4 = Chain(conv_norm((1, 1), 128, 320)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2, branch3),
                                branch4, inputscale(scale; activation = relu)), +)
end

function mixed_6a()
    branch1 = Chain(conv_norm((3, 3), 320, 384; stride = 2)...)
    branch2 = Chain(conv_norm((1, 1), 320, 256)...,
                    conv_norm((3, 3), 256, 256; pad = 1)...,
                    conv_norm((3, 3), 256, 384; stride = 2)...)
    branch3 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3)
end

function block17(scale = 1.0f0)
    branch1 = Chain(conv_norm((1, 1), 1088, 192)...)
    branch2 = Chain(conv_norm((1, 1), 1088, 128)...,
                    conv_norm((1, 7), 128, 160; pad = (0, 3))...,
                    conv_norm((7, 1), 160, 192; pad = (3, 0))...)
    branch3 = Chain(conv_norm((1, 1), 384, 1088)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2),
                                branch3, inputscale(scale; activation = relu)), +)
end

function mixed_7a()
    branch1 = Chain(conv_norm((1, 1), 1088, 256)...,
                    conv_norm((3, 3), 256, 384; stride = 2)...)
    branch2 = Chain(conv_norm((1, 1), 1088, 256)...,
                    conv_norm((3, 3), 256, 288; stride = 2)...)
    branch3 = Chain(conv_norm((1, 1), 1088, 256)...,
                    conv_norm((3, 3), 256, 288; pad = 1)...,
                    conv_norm((3, 3), 288, 320; stride = 2)...)
    branch4 = MaxPool((3, 3); stride = 2)
    return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function block8(scale = 1.0f0; activation = identity)
    branch1 = Chain(conv_norm((1, 1), 2080, 192)...)
    branch2 = Chain(conv_norm((1, 1), 2080, 192)...,
                    conv_norm((1, 3), 192, 224; pad = (0, 1))...,
                    conv_norm((3, 1), 224, 256; pad = (1, 0))...)
    branch3 = Chain(conv_norm((1, 1), 448, 2080)...)
    return SkipConnection(Chain(Parallel(cat_channels, branch1, branch2),
                                branch3, inputscale(scale; activation)), +)
end

"""
    inceptionresnetv2(; inchannels::Integer = 3, dropout_rate = 0.0, nclasses::Integer = 1000)

Creates an InceptionResNetv2 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `inchannels`: number of input channels.
  - `dropout_rate`: rate of dropout in classifier head.
  - `nclasses`: the number of output classes.
"""
function inceptionresnetv2(; dropout_rate = 0.0, inchannels::Integer = 3,
                           nclasses::Integer = 1000)
    backbone = Chain(conv_norm((3, 3), inchannels, 32; stride = 2)...,
                     conv_norm((3, 3), 32, 32)...,
                     conv_norm((3, 3), 32, 64; pad = 1)...,
                     MaxPool((3, 3); stride = 2),
                     conv_norm((3, 3), 64, 80)...,
                     conv_norm((3, 3), 80, 192)...,
                     MaxPool((3, 3); stride = 2),
                     mixed_5b(),
                     [block35(0.17f0) for _ in 1:10]...,
                     mixed_6a(),
                     [block17(0.10f0) for _ in 1:20]...,
                     mixed_7a(),
                     [block8(0.20f0) for _ in 1:9]...,
                     block8(; activation = relu),
                     conv_norm((1, 1), 2080, 1536)...)
    return Chain(backbone, create_classifier(1536, nclasses; dropout_rate))
end

"""
    InceptionResNetv2(; pretrain::Bool = false, inchannels::Integer = 3, 
                      nclasses::Integer = 1000)

Creates an InceptionResNetv2 model.
([reference](https://arxiv.org/abs/1602.07261))

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels.
  - `nclasses`: the number of output classes.

!!! warning
    
    `InceptionResNetv2` does not currently support pretrained weights.
"""
struct InceptionResNetv2
    layers::Any
end
@functor InceptionResNetv2

function InceptionResNetv2(; pretrain::Bool = false, inchannels::Integer = 3,
                           nclasses::Integer = 1000)
    layers = inceptionresnetv2(; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "InceptionResNetv2")
    end
    return InceptionResNetv2(layers)
end

(m::InceptionResNetv2)(x) = m.layers(x)

backbone(m::InceptionResNetv2) = m.layers[1]
classifier(m::InceptionResNetv2) = m.layers[2]
