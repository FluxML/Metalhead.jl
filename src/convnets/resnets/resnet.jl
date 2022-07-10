const resnet_shortcuts = Dict(18 => [:A, :B, :B, :B],
                              34 => [:A, :B, :B, :B],
                              50 => [:B, :B, :B, :B],
                              101 => [:B, :B, :B, :B],
                              152 => [:B, :B, :B, :B])

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

(m::ResNet)(x) = m.layers(x)

function ResNet(depth::Integer; pretrain = false, inchannels = 3, nclasses = 1000)
    @assert depth in [18, 34, 50, 101, 152]
    "Invalid depth. Must be one of [18, 34, 50, 101, 152]"
    layers = resnet(resnet_config[depth]..., resnet_shortcuts[depth]; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("ResNet", depth))
    end
    return ResNet(layers)
end
