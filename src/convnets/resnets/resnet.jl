"""
    ResNet(depth::Integer; pretrain = false, inchannels = 3, nclasses = 1000)

Creates a ResNet model with the specified depth.
((reference)[https://arxiv.org/abs/1512.03385])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: the number of output classes

!!! warning
    
    `ResNet` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](#).
"""
struct ResNet
    layers::Any
end
@functor ResNet

function ResNet(depth::Integer; pretrain = false, inchannels = 3, nclasses = 1000)
    _checkconfig(depth, keys(resnet_configs))
    layers = resnet(resnet_configs[depth]..., resnet_shortcuts[depth]; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("ResNet", depth))
    end
    return ResNet(layers)
end

(m::ResNet)(x) = m.layers(x)

backbone(m::ResNet) = m.layers[1]
classifier(m::ResNet) = m.layers[2]
