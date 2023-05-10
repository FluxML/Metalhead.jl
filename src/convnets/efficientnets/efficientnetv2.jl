# block configs for EfficientNetv2
# k: kernel size
# c: output channels
# e: expansion ratio
# s: stride
# n: number of repeats
# r: reduction ratio for squeeze-excite layer - specified only for `mbconv`
# a: activation function
# Data organised as (f, k, c, e, s, n, (r,) a) for each stage
const EFFNETV2_CONFIGS = Dict(:small => [(fused_mbconv, 3, 24, 1, 1, 2, swish),
                                  (fused_mbconv, 3, 48, 4, 2, 4, swish),
                                  (fused_mbconv, 3, 64, 4, 2, 4, swish),
                                  (mbconv, 3, 128, 4, 2, 6, 4, swish),
                                  (mbconv, 3, 160, 6, 1, 9, 4, swish),
                                  (mbconv, 3, 256, 6, 2, 15, 4, swish)],
                              :medium => [(fused_mbconv, 3, 24, 1, 1, 3, swish),
                                  (fused_mbconv, 3, 48, 4, 2, 5, swish),
                                  (fused_mbconv, 3, 80, 4, 2, 5, swish),
                                  (mbconv, 3, 160, 4, 2, 7, 4, swish),
                                  (mbconv, 3, 176, 6, 1, 14, 4, swish),
                                  (mbconv, 3, 304, 6, 2, 18, 4, swish),
                                  (mbconv, 3, 512, 6, 1, 5, 4, swish)],
                              :large => [(fused_mbconv, 3, 32, 1, 1, 4, swish),
                                  (fused_mbconv, 3, 64, 4, 2, 7, swish),
                                  (fused_mbconv, 3, 96, 4, 2, 7, swish),
                                  (mbconv, 3, 192, 4, 2, 10, 4, swish),
                                  (mbconv, 3, 224, 6, 1, 19, 4, swish),
                                  (mbconv, 3, 384, 6, 2, 25, 4, swish),
                                  (mbconv, 3, 640, 6, 1, 7, 4, swish)],
                              :xlarge => [(fused_mbconv, 3, 32, 1, 1, 4, swish),
                                  (fused_mbconv, 3, 64, 4, 2, 8, swish),
                                  (fused_mbconv, 3, 96, 4, 2, 8, swish),
                                  (mbconv, 3, 192, 4, 2, 16, 4, swish),
                                  (mbconv, 3, 384, 6, 1, 24, 4, swish),
                                  (mbconv, 3, 512, 6, 2, 32, 4, swish),
                                  (mbconv, 3, 768, 6, 1, 8, 4, swish)])

"""
    efficientnetv2(config::Symbol; norm_layer = BatchNorm, stochastic_depth_prob = 0.2,
                   dropout_prob = nothing, inchannels::Integer = 3, nclasses::Integer = 1000)

Create an EfficientNetv2 model. ([reference](https://arxiv.org/abs/2104.00298)).

# Arguments

  - `config`: size of the network (one of `[:small, :medium, :large, :xlarge]`)
  - `norm_layer`: normalization layer to use.
  - `stochastic_depth_prob`: probability of stochastic depth. Set to `nothing` to disable
    stochastic depth.
  - `dropout_prob`: probability of dropout in the classifier head. Set to `nothing` to disable
    dropout.
  - `inchannels`: number of input channels.
  - `nclasses`: number of output classes.
"""
function efficientnetv2(config::Symbol; norm_layer = BatchNorm, stochastic_depth_prob = 0.2,
                        dropout_prob = nothing, inchannels::Integer = 3,
                        nclasses::Integer = 1000)
    _checkconfig(config, keys(EFFNETV2_CONFIGS))
    block_configs = EFFNETV2_CONFIGS[config]
    return build_invresmodel((1, 1), block_configs; activation = swish, norm_layer,
                             inplanes = block_configs[1][3], headplanes = 1280,
                             stochastic_depth_prob, dropout_prob, inchannels, nclasses)
end

"""
    EfficientNetv2(config::Symbol; pretrain::Bool = false, inchannels::Integer = 3,
                   nclasses::Integer = 1000)

Create an EfficientNetv2 model ([reference](https://arxiv.org/abs/2104.00298)).

# Arguments

  - `config`: size of the network (one of `[:small, :medium, :large, :xlarge]`)
  - `pretrain`: whether to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels
  - `nclasses`: number of output classes

!!! warning
    
    `EfficientNetv2` does not currently support pretrained weights.

See also [`efficientnet`](#).
"""
struct EfficientNetv2
    layers::Any
end
@functor EfficientNetv2

function EfficientNetv2(config::Symbol; pretrain::Bool = false,
                        inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = efficientnetv2(config; inchannels, nclasses)
    model = EfficientNetv2(layers)
    if pretrain
        loadpretrain!(model, string("efficientnet_v2_", config))
    end
    return model
end

(m::EfficientNetv2)(x) = m.layers(x)

backbone(m::EfficientNetv2) = m.layers[1]
classifier(m::EfficientNetv2) = m.layers[2]
