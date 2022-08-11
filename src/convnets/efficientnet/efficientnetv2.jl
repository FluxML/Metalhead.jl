"""
    efficientnetv2(config::AbstractVector{<:Tuple}; max_width::Integer = 1792,
                   width_mult::Real = 1.0, inchannels::Integer = 3,
                   nclasses::Integer = 1000)

Create an EfficientNetv2 model ([reference](https://arxiv.org/abs/2104.00298)).

# Arguments

  - `config`: configuration for each inverted residual block,
    given as a vector of tuples with elements:
    
      + `t`: expansion factor of the block
      + `c`: output channels of the block (will be scaled by width_mult)
      + `n`: number of block repetitions
      + `s`: kernel stride in the block except the first block of each stage
      + `r`: reduction factor of the squeeze-excite layer

  - `max_width`: The maximum number of feature maps in any layer of the network
  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper)
  - `inchannels`: number of input channels
  - `nclasses`: number of output classes
"""
function efficientnetv2(config::AbstractVector{<:Tuple}; max_width::Integer = 1792,
                        width_mult::Real = 1.0, inchannels::Integer = 3,
                        nclasses::Integer = 1000)
    # building first layer
    inplanes = _round_channels(24 * width_mult, 8)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes, swish; pad = 1, stride = 2,
                      bias = false))
    # building inverted residual blocks
    for (t, inplanes, outplanes, n, s, reduction) in config
        explanes = _round_channels(inplanes * t, 8)
        for i in 1:n
            stride = i == 1 ? s : 1
            if isnothing(reduction)
                push!(layers,
                      fused_mbconv((3, 3), inplanes, explanes, outplanes, swish; stride))
            else
                inplanes = _round_channels(inplanes * width_mult, 8)
                outplanes = _round_channels(outplanes * width_mult, 8)
                push!(layers,
                      mbconv((3, 3), inplanes, explanes, outplanes, swish; stride))
            end
            inplanes = outplanes
        end
    end
    # building last layers
    outplanes = width_mult > 1 ? _round_channels(max_width * width_mult, 8) :
                max_width
    append!(layers, conv_norm((1, 1), inplanes, outplanes, swish; bias = false))
    return Chain(Chain(layers...), create_classifier(outplanes, nclasses))
end

# block configs for EfficientNetv2
const EFFNETV2_CONFIGS = Dict(:small => [
                                  (1, 24, 24, 2, 1, nothing),
                                  (4, 24, 48, 4, 2, nothing),
                                  (4, 48, 64, 4, 2, nothing),
                                  (4, 64, 128, 6, 2, 4),
                                  (6, 128, 160, 9, 1, 4),
                                  (6, 160, 256, 15, 2, 4)],
                              :medium => [
                                  (1, 24, 24, 3, 1, nothing),
                                  (4, 24, 48, 5, 2, nothing),
                                  (4, 48, 80, 5, 2, nothing),
                                  (4, 80, 160, 7, 2, 4),
                                  (6, 160, 176, 14, 1, 4),
                                  (6, 176, 304, 18, 2, 4),
                                  (6, 304, 512, 5, 1, 4)],
                              :large => [
                                  (1, 32, 32, 4, 1, nothing),
                                  (4, 32, 64, 7, 2, nothing),
                                  (4, 64, 96, 7, 2, nothing),
                                  (4, 96, 192, 10, 2, 4),
                                  (6, 192, 224, 19, 1, 4),
                                  (6, 224, 384, 25, 2, 4),
                                  (6, 384, 640, 7, 1, 4)],
                              :xlarge => [
                                  (1, 32, 32, 4, 1, nothing),
                                  (4, 32, 64, 8, 2, nothing),
                                  (4, 64, 96, 8, 2, nothing),
                                  (4, 96, 192, 16, 2, 4),
                                  (6, 192, 256, 24, 1, 4),
                                  (6, 256, 512, 32, 2, 4),
                                  (6, 512, 640, 8, 1, 4)])

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
