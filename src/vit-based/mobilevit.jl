function mobilevit(imsize:Dims{2} = (256, 256); inchannels::Integer = 3,)
    return Chain(Conv((3,3)))
end

struct MobileViT
    layers::Any
end
@functor MobileViT

const MOBILE_VIT_CONFIGS = Dict(:small => (width_multiplier=2, ffn_multiplier=2, swish),
                                :x-small => (),
                                :xx-small => ())

"""
    MobileViT(config::Symbol; imsize::Dims{2}=(256, 256), patch_size::Dims{2}=(16,16),
        pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a mobilevit model
([reference](https://arxiv.org/abs/2110.02178))

# Arguments

    - `config`: the model configuration, one of

"""
function MobileViT(config::Symbol; imsize::Dims{2}=(256, 256), patch_size::Dims{2}=(16,16),
    pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, keys(MOBILE_VIT_CONFIGS))

    layers = mobilevit(imsize; inchannels)

    return MobileViT(layers)
end

(m::MobileViT)(x) = m.layers(x)

backbone(m::ViT) = m.layers[1]
classifier(m::ViT) = m.layers[2]