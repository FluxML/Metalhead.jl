"""
    inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_3x3, pool_proj)

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
  - `batchnorm`: set to `true` to include batch normalization after each convolution
  - `bias`: set to `true` to use bias in the convolution layers.
"""
function inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj,
                        batchnorm, bias)
    branch1 = Chain(basic_conv_bn((1, 1), inplanes, out_1x1; batchnorm = batchnorm,
                                  bias = bias)...)
    branch2 = Chain(basic_conv_bn((1, 1), inplanes, red_3x3; batchnorm = batchnorm,
                                  bias = bias)...,
                    basic_conv_bn((3, 3), red_3x3, out_3x3; batchnorm = batchnorm,
                                  bias = bias, pad = 1)...)
    branch3 = Chain(basic_conv_bn((1, 1), inplanes, red_5x5; batchnorm = batchnorm,
                                  bias = bias)...,
                    basic_conv_bn((5, 5), red_5x5, out_5x5; batchnorm = batchnorm,
                                  bias = bias, pad = 2)...)
    branch4 = Chain(MaxPool((3, 3); stride = 1, pad = 1),
                    basic_conv_bn((1, 1), inplanes, pool_proj; batchnorm = batchnorm,
                                  bias = bias)...)
    return Parallel(cat_channels,
                    branch1, branch2, branch3, branch4)
end

"""
    googlenet(; dropout_prob = 0.4, inchannels::Integer = 3, nclasses::Integer = 1000)

Create an Inception-v1 model (commonly referred to as GoogLeNet)
([reference](https://arxiv.org/abs/1409.4842v1)).

# Arguments

  - `dropout_prob`: the dropout probability in the classifier head. Set to `nothing` to disable dropout.
  - `inchannels`: the number of input channels
  - `nclasses`: the number of output classes
  - `batchnorm`: set to `true` to include batch normalization after each convolution
  - `bias`: set to `true` to use bias in the convolution layers
"""
function googlenet(; dropout_prob = 0.4, inchannels::Integer = 3, nclasses::Integer = 1000,
                   batchnorm::Bool = false, bias::Bool = true)
    backbone = Chain(basic_conv_bn((7, 7), inchannels, 64; batchnorm = batchnorm,
                                   stride = 2, pad = 3, bias = bias)...,
                     MaxPool((3, 3); stride = 2, pad = 1),
                     basic_conv_bn((1, 1), 64, 64; batchnorm = batchnorm, bias = bias)...,
                     basic_conv_bn((3, 3), 64, 192; batchnorm = batchnorm, pad = 1,
                                   bias = bias)...,
                     MaxPool((3, 3); stride = 2, pad = 1),
                     inceptionblock(192, 64, 96, 128, 16, 32, 32, batchnorm, bias),
                     inceptionblock(256, 128, 128, 192, 32, 96, 64, batchnorm, bias),
                     MaxPool((3, 3); stride = 2, pad = 1),
                     inceptionblock(480, 192, 96, 208, 16, 48, 64, batchnorm, bias),
                     inceptionblock(512, 160, 112, 224, 24, 64, 64, batchnorm, bias),
                     inceptionblock(512, 128, 128, 256, 24, 64, 64, batchnorm, bias),
                     inceptionblock(512, 112, 144, 288, 32, 64, 64, batchnorm, bias),
                     inceptionblock(528, 256, 160, 320, 32, 128, 128, batchnorm, bias),
                     MaxPool((3, 3); stride = 2, pad = 1),
                     inceptionblock(832, 256, 160, 320, 32, 128, 128, batchnorm, bias),
                     inceptionblock(832, 384, 192, 384, 48, 128, 128, batchnorm, bias))
    return Chain(backbone, create_classifier(1024, nclasses; dropout_prob))
end

"""
    GoogLeNet(; pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)

Create an Inception-v1 model (commonly referred to as `GoogLeNet`)
([reference](https://arxiv.org/abs/1409.4842v1)).

# Arguments

  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `nclasses`: the number of output classes
  - `batchnorm`: set to `true` to use batch normalization after each convolution
  - `bias`: set to `true` to use bias in the convolution layers

!!! warning
    
    `GoogLeNet` does not currently support pretrained weights.

See also [`Metalhead.googlenet`](@ref).
"""
struct GoogLeNet
    layers::Any
end
@functor GoogLeNet

function GoogLeNet(; pretrain::Bool = false, inchannels::Integer = 3,
                   nclasses::Integer = 1000, batchnorm::Bool = false, bias::Bool = true)
    layers = googlenet(; inchannels, nclasses, batchnorm, bias)
    if pretrain
        loadpretrain!(layers, "GoogLeNet")
    end
    return GoogLeNet(layers)
end

(m::GoogLeNet)(x) = m.layers(x)

backbone(m::GoogLeNet) = m.layers[1]
classifier(m::GoogLeNet) = m.layers[2]
