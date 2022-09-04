"""
    Res2Net(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
            base_width::Integer = 26, inchannels::Integer = 3,
            nclasses::Integer = 1000)

Creates a Res2Net model with the specified depth, scale, and base width.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the Res2Net model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `scale`: the number of feature groups in the block. See the
    [paper](https://arxiv.org/abs/1904.01169) for more details.
  - `base_width`: the number of feature maps in each group.
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes

!!! warning
    
    `Res2Net` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](@ref).
"""
struct Res2Net
    layers::Any
end
@functor Res2Net

function Res2Net(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
                 base_width::Integer = 26, inchannels::Integer = 3,
                 nclasses::Integer = 1000)
    _checkconfig(depth, keys(LRESNET_CONFIGS))
    layers = resnet(bottle2neck, LRESNET_CONFIGS[depth][2]; base_width, scale,
                    inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("Res2Net", depth, "_", base_width, "x", scale))
    end
    return Res2Net(layers)
end

(m::Res2Net)(x) = m.layers(x)

backbone(m::Res2Net) = m.layers[1]
classifier(m::Res2Net) = m.layers[2]

"""
    Res2NeXt(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
             base_width::Integer = 4, cardinality::Integer = 8,
             inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a Res2NeXt model with the specified depth, scale, base width and cardinality.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the Res2Net model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `scale`: the number of feature groups in the block. See the
    [paper](https://arxiv.org/abs/1904.01169) for more details.
  - `base_width`: the number of feature maps in each group.
  - `cardinality`: the number of groups in the 3x3 convolutions.
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes

!!! warning
    
    `Res2NeXt` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](@ref).
"""
struct Res2NeXt
    layers::Any
end
@functor Res2NeXt

function Res2NeXt(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
                  base_width::Integer = 4, cardinality::Integer = 8,
                  inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(depth, keys(LRESNET_CONFIGS))
    layers = resnet(bottle2neck, LRESNET_CONFIGS[depth][2]; base_width, scale,
                    cardinality, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers,
                      string("Res2NeXt", depth, "_", base_width, "x", cardinality,
                             "x", scale))
    end
    return Res2NeXt(layers)
end

(m::Res2NeXt)(x) = m.layers(x)

backbone(m::Res2NeXt) = m.layers[1]
classifier(m::Res2NeXt) = m.layers[2]
