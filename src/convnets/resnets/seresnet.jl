"""
    SEResNet(depth::Integer; pretrain = false, inchannels = 3, nclasses = 1000)

Creates a SEResNet model with the specified depth.
((reference)[https://arxiv.org/pdf/1709.01507.pdf])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes

!!! warning
    
    `SEResNet` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](#).
"""
struct SEResNet
    layers::Any
end
@functor SEResNet

(m::SEResNet)(x) = m.layers(x)

function SEResNet(depth::Integer; pretrain = false, inchannels = 3, nclasses = 1000)
    _checkconfig(depth, keys(resnet_configs))
    layers = resnet(resnet_configs[depth]...; inchannels, nclasses,
                    attn_fn = squeeze_excite)
    if pretrain
        loadpretrain!(layers, string("SEResNet", depth))
    end
    return SEResNet(layers)
end

backbone(m::SEResNet) = m.layers[1]
classifier(m::SEResNet) = m.layers[2]

"""
    SEResNeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4,
              inchannels = 3, nclasses = 1000)

Creates a SEResNeXt model with the specified depth, cardinality, and base width.
((reference)[https://arxiv.org/pdf/1709.01507.pdf])

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `cardinality`: the number of groups to be used in the 3x3 convolution in each block.
  - `base_width`: the number of feature maps in each group.
  - `inchannels`: the number of input channels
  - `nclasses`: the number of output classes

!!! warning
    
    `SEResNeXt` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](#).
"""
struct SEResNeXt
    layers::Any
end
@functor SEResNeXt

(m::SEResNeXt)(x) = m.layers(x)

function SEResNeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4,
                   inchannels = 3, nclasses = 1000)
    _checkconfig(depth, sort(collect(keys(resnet_configs)))[3:end])
    layers = resnet(resnet_configs[depth]...; inchannels, nclasses, cardinality, base_width,
                    attn_fn = squeeze_excite)
    if pretrain
        loadpretrain!(layers, string("SEResNeXt", depth, "_", cardinality, "x", base_width))
    end
    return SEResNeXt(layers)
end

backbone(m::SEResNeXt) = m.layers[1]
classifier(m::SEResNeXt) = m.layers[2]
