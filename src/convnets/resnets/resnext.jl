"""
    ResNeXt(depth::Integer; pretrain = false, cardinality = 32,
            base_width = 4, inchannels = 3, nclasses = 1000)

Creates a ResNeXt model with the specified depth, cardinality, and base width.
((reference)[https://arxiv.org/abs/1611.05431])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `cardinality`: the number of groups to be used in the 3x3 convolution in each block.
  - `base_width`: the number of feature maps in each group.
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes

!!! warning
    
    `ResNeXt` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](#).
"""
struct ResNeXt
    layers::Any
end
@functor ResNeXt

(m::ResNeXt)(x) = m.layers(x)

function ResNeXt(depth::Integer; pretrain = false, cardinality = 32,
                 base_width = 4, inchannels = 3, nclasses = 1000)
    @assert depth in [50, 101, 152]
    "Invalid depth. Must be one of [50, 101, 152]"
    layers = resnet(resnet_config[depth]...; inchannels, nclasses,
                    block_args = (; cardinality, base_width))
    if pretrain
        loadpretrain!(layers, string("ResNeXt", depth, "_", cardinality, "x", base_width))
    end
    return ResNeXt(layers)
end

backbone(m::ResNeXt) = m.layers[1]
classifier(m::ResNeXt) = m.layers[2]
