"""
    SEResNet(depth::Integer; pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a SEResNet model with the specified depth.
([reference](https://arxiv.org/pdf/1709.01507.pdf))

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes

!!! warning
    
    `SEResNet` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](@ref).
"""
struct SEResNet
    layers::Any
end
@functor SEResNet

function SEResNet(depth::Integer; pretrain::Bool = false, inchannels::Integer = 3,
                  nclasses::Integer = 1000)
    _checkconfig(depth, keys(RESNET_CONFIGS))
    layers = resnet(RESNET_CONFIGS[depth]...; inchannels, nclasses,
                    attn_fn = squeeze_excite)
    model = SEResNet(layers)
    if pretrain
        artifact_name = "seresnet$(depth)"
        loadpretrain!(model, artifact_name)
    end
    return model
end

(m::SEResNet)(x) = m.layers(x)

backbone(m::SEResNet) = m.layers[1]
classifier(m::SEResNet) = m.layers[2]

"""
    SEResNeXt(depth::Integer; pretrain::Bool = false, cardinality::Integer = 32,
              base_width::Integer = 4, inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a SEResNeXt model with the specified depth, cardinality, and base width.
([reference](https://arxiv.org/pdf/1709.01507.pdf))

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `cardinality`: the number of groups to be used in the 3x3 convolution in each block.
  - `base_width`: the number of feature maps in each group.
  - `inchannels`: the number of input channels
  - `nclasses`: the number of output classes

!!! warning
    
    `SEResNeXt` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](@ref).
"""
struct SEResNeXt
    layers::Any
end
@functor SEResNeXt

function SEResNeXt(depth::Integer; pretrain::Bool = false, cardinality::Integer = 32,
                   base_width::Integer = 4, inchannels::Integer = 3,
                   nclasses::Integer = 1000)
    _checkconfig(depth, keys(LRESNET_CONFIGS))
    layers = resnet(LRESNET_CONFIGS[depth]...; inchannels, nclasses, cardinality,
                    base_width,
                    attn_fn = squeeze_excite)
    model = SEResNeXt(layers)
    if pretrain
        loadpretrain!(model, string("seresnext", depth, "_", cardinality, "x", base_width))
    end
    return model
end

(m::SEResNeXt)(x) = m.layers(x)

backbone(m::SEResNeXt) = m.layers[1]
classifier(m::SEResNeXt) = m.layers[2]
