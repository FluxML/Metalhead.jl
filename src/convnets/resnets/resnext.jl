"""
    ResNeXt(depth::Integer; pretrain::Bool = false, cardinality::Integer = 32,
            base_width::Integer = 4, inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a ResNeXt model with the specified depth, cardinality, and base width.
([reference](https://arxiv.org/abs/1611.05431))

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the ResNet model.

  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet.
    Supported configurations are:
    
      + depth 50, cardinality of 32 and base width of 4.
      + depth 101, cardinality of 32 and base width of 8.
      + depth 101, cardinality of 64 and base width of 4.
  - `cardinality`: the number of groups to be used in the 3x3 convolution in each block.
  - `base_width`: the number of feature maps in each group.
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes

Advanced users who want more configuration options will be better served by using [`resnet`](@ref).
"""
struct ResNeXt
    layers::Any
end
@functor ResNeXt

function ResNeXt(depth::Integer; pretrain::Bool = false, cardinality::Integer = 32,
                 base_width::Integer = 4, inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(depth, keys(LRESNET_CONFIGS))
    layers = resnet(LRESNET_CONFIGS[depth]...; inchannels, nclasses, cardinality,
                    base_width)
    model = ResNeXt(layers)
    if pretrain
        artifact_name = string("resnext", depth, "_", cardinality, "x", base_width, "d")
        if depth == 50 && cardinality == 32 && base_width == 4
          artifact_name *= "-IMAGENET1K_V2"
        elseif depth == 101 && cardinality == 32 && base_width == 8
          artifact_name *= "-IMAGENET1K_V2"
        elseif depth == 101 && cardinality == 64 && base_width == 4
          artifact_name *= "-IMAGENET1K_V2"
        end
        loadpretrain!(model, artifact_name)
    end
    return model
end

(m::ResNeXt)(x) = m.layers(x)

backbone(m::ResNeXt) = m.layers[1]
classifier(m::ResNeXt) = m.layers[2]
