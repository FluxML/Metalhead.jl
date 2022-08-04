function efficientnetv2(config::AbstractVector{NTuple{5, Int}}; max_width::Integer = 1792,
                        width_mult::Real = 1.0, inchannels::Integer = 3,
                        nclasses::Integer = 1000)
    # building first layer
    inplanes = _round_channels(24 * width_mult, 8)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes, swish; pad = 1, stride = 2,
                      bias = false))
    # building inverted residual blocks
    for (t, c, n, s, r) in config
        outplanes = _round_channels(c * width_mult, 8)
        for i in 1:n
            push!(layers,
                  invertedresidual((3, 3), inplanes, outplanes, swish; expansion = t,
                                   stride = i == 1 ? s : 1,
                                   reduction = r == 1 ? 4 : nothing))
            inplanes = outplanes
        end
    end
    # building last layers
    outplanes = width_mult > 1 ? _round_channels(max_width * width_mult, 8) :
                max_width
    append!(layers, conv_norm((1, 1), inplanes, outplanes, swish; bias = false))
    return Chain(Chain(layers...), create_classifier(outplanes, nclasses))
end

const EFFNETV2_CONFIGS = Dict(:small => [# t, c, n, s, SE
                                  (1, 24, 2, 1, 0),
                                  (4, 48, 4, 2, 0),
                                  (4, 64, 4, 2, 0),
                                  (4, 128, 6, 2, 1),
                                  (6, 160, 9, 1, 1),
                                  (6, 256, 15, 2, 1)],
                              :medium => [# t, c, n, s, SE
                                  (1, 24, 3, 1, 0),
                                  (4, 48, 5, 2, 0),
                                  (4, 80, 5, 2, 0),
                                  (4, 160, 7, 2, 1),
                                  (6, 176, 14, 1, 1),
                                  (6, 304, 18, 2, 1),
                                  (6, 512, 5, 1, 1)],
                              :large => [# t, c, n, s, SE
                                  (1, 32, 4, 1, 0),
                                  (4, 64, 8, 2, 0),
                                  (4, 96, 8, 2, 0),
                                  (4, 192, 16, 2, 1),
                                  (6, 256, 24, 1, 1),
                                  (6, 512, 32, 2, 1),
                                  (6, 640, 8, 1, 1)],
                              :xlarge => [# t, c, n, s, SE
                                  (1, 32, 4, 1, 0),
                                  (4, 64, 8, 2, 0),
                                  (4, 96, 8, 2, 0),
                                  (4, 192, 16, 2, 1),
                                  (6, 256, 24, 1, 1),
                                  (6, 512, 32, 2, 1),
                                  (6, 640, 8, 1, 1)])

struct EfficientNetv2
    layers::Any
end
@functor EfficientNetv2

function EfficientNetv2(config::Symbol; pretrain::Bool = false, width_mult::Real = 1,
                        inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, sort(collect(keys(EFFNETV2_CONFIGS))))
    layers = efficientnetv2(EFFNETV2_CONFIGS[config]; width_mult, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("efficientnetv2"))
    end
    return EfficientNetv2(layers)
end

(m::EfficientNetv2)(x) = m.layers(x)

backbone(m::EfficientNetv2) = m.layers[1]
classifier(m::EfficientNetv2) = m.layers[2]
