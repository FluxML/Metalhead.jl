# block configs for EfficientNetv2
const EFFNETV2_CONFIGS = Dict(:small => [
                                  FusedMBConvConfig(3, 24, 24, 1, 1, 2),
                                  FusedMBConvConfig(3, 24, 48, 4, 2, 4),
                                  FusedMBConvConfig(3, 48, 64, 4, 2, 4),
                                  MBConvConfig(3, 64, 128, 4, 2, 6),
                                  MBConvConfig(3, 128, 160, 6, 1, 9),
                                  MBConvConfig(3, 160, 256, 6, 2, 15)],
                              :medium => [
                                  FusedMBConvConfig(3, 24, 24, 1, 1, 3),
                                  FusedMBConvConfig(3, 24, 48, 4, 2, 5),
                                  FusedMBConvConfig(3, 48, 80, 4, 2, 5),
                                  MBConvConfig(3, 80, 160, 4, 2, 7),
                                  MBConvConfig(3, 160, 176, 6, 1, 14),
                                  MBConvConfig(3, 176, 304, 6, 2, 18),
                                  MBConvConfig(3, 304, 512, 6, 1, 5)],
                              :large => [
                                  FusedMBConvConfig(3, 32, 32, 1, 1, 4),
                                  FusedMBConvConfig(3, 32, 64, 4, 2, 7),
                                  FusedMBConvConfig(3, 64, 96, 4, 2, 7),
                                  MBConvConfig(3, 96, 192, 4, 2, 10),
                                  MBConvConfig(3, 192, 224, 6, 1, 19),
                                  MBConvConfig(3, 224, 384, 6, 2, 25),
                                  MBConvConfig(3, 384, 640, 6, 1, 7)],
                              :xlarge => [
                                  FusedMBConvConfig(3, 32, 32, 1, 1, 4),
                                  FusedMBConvConfig(3, 32, 64, 4, 2, 8),
                                  FusedMBConvConfig(3, 64, 96, 4, 2, 8),
                                  MBConvConfig(3, 96, 192, 4, 2, 16),
                                  MBConvConfig(3, 192, 224, 6, 1, 24),
                                  MBConvConfig(3, 384, 512, 6, 2, 32),
                                  MBConvConfig(3, 512, 768, 6, 1, 8)])

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
    layers = efficientnet(EFFNETV2_CONFIGS[config]; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("efficientnetv2"))
    end
    return EfficientNetv2(layers)
end

(m::EfficientNetv2)(x) = m.layers(x)

backbone(m::EfficientNetv2) = m.layers[1]
classifier(m::EfficientNetv2) = m.layers[2]
