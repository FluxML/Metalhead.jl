# block configs for EfficientNetv2
# data organised as (k, i, o, e, s, n)
const EFFNETV2_CONFIGS = Dict(:small => [
                                  (3, 24, 24, 1, 1, 2),
                                  (3, 24, 48, 4, 2, 4),
                                  (3, 48, 64, 4, 2, 4),
                                  (3, 64, 128, 4, 2, 6),
                                  (3, 128, 160, 6, 1, 9),
                                  (3, 160, 256, 6, 2, 15)],
                              :medium => [
                                  (3, 24, 24, 1, 1, 3),
                                  (3, 24, 48, 4, 2, 5),
                                  (3, 48, 80, 4, 2, 5),
                                  (3, 80, 160, 4, 2, 7),
                                  (3, 160, 176, 6, 1, 14),
                                  (3, 176, 304, 6, 2, 18),
                                  (3, 304, 512, 6, 1, 5)],
                              :large => [
                                  (3, 32, 32, 1, 1, 4),
                                  (3, 32, 64, 4, 2, 7),
                                  (3, 64, 96, 4, 2, 7),
                                  (3, 96, 192, 4, 2, 10),
                                  (3, 192, 224, 6, 1, 19),
                                  (3, 224, 384, 6, 2, 25),
                                  (3, 384, 640, 6, 1, 7)],
                              :xlarge => [
                                  (3, 32, 32, 1, 1, 4),
                                  (3, 32, 64, 4, 2, 8),
                                  (3, 64, 96, 4, 2, 8),
                                  (3, 96, 192, 4, 2, 16),
                                  (3, 192, 224, 6, 1, 24),
                                  (3, 384, 512, 6, 2, 32),
                                  (3, 512, 768, 6, 1, 8)])

"""
    EfficientNetv2(config::Symbol; pretrain::Bool = false, width_mult::Real = 1,
                   inchannels::Integer = 3, nclasses::Integer = 1000)

Create an EfficientNetv2 model ([reference](https://arxiv.org/abs/2104.00298)).

# Arguments

  - `config`: size of the network (one of `[:small, :medium, :large, :xlarge]`)
  - `pretrain`: whether to load the pre-trained weights for ImageNet
  - `width_mult`: Controls the number of output feature maps in each block (with 1
    being the default in the paper)
  - `inchannels`: number of input channels
  - `nclasses`: number of output classes
"""
struct EfficientNetv2
    layers::Any
end
@functor EfficientNetv2

function EfficientNetv2(config::Symbol; pretrain::Bool = false,
                        inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, sort(collect(keys(EFFNETV2_CONFIGS))))
    layers = efficientnet(EFFNETV2_CONFIGS[config],
                          vcat(fill(fused_mbconv_builder, 3),
                               fill(mbconv_builder, length(EFFNETV2_CONFIGS[config]) - 3));
                          headplanes = 1280, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("efficientnetv2"))
    end
    return EfficientNetv2(layers)
end

(m::EfficientNetv2)(x) = m.layers(x)

backbone(m::EfficientNetv2) = m.layers[1]
classifier(m::EfficientNetv2) = m.layers[2]
