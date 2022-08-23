"""
    mobilenetv3(configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
                max_width::Integer = 1024, dropout_rate = 0.2,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv3 model.
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments

  - `configs`: a "list of tuples" configuration for each layer that details:
    
      + `k::Integer` - The size of the convolutional kernel
      + `c::Float` - The multiplier factor for deciding the number of feature maps in the hidden layer
      + `t::Integer` - The number of output feature maps for a given block
      + `r::Integer` - The reduction factor (`>= 1` or `nothing` to skip) for squeeze and excite layers
      + `s::Integer` - The stride of the convolutional kernel
      + `a` - The activation function used in the bottleneck (typically `hardswish` or `relu`)

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper; this is usually a value between 0.1 and 1.4.)
  - `max_width`: The maximum number of feature maps in any layer of the network
  - `dropout_rate`: The dropout rate to use in the classifier head. Set to `nothing` to disable.
  - `inchannels`: The number of input channels.
  - `nclasses`: the number of output classes
"""
function mobilenetv3(configs::AbstractVector{<:Tuple}; width_mult::Real = 1,
                     max_width::Integer = 1024, dropout_rate = 0.2,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    # building first layer
    inplanes = _round_channels(16 * width_mult, 8)
    layers = []
    append!(layers,
            conv_norm((3, 3), inchannels, inplanes, hardswish; stride = 2, pad = 1))
    # building inverted residual blocks
    get_layers, block_repeats = mbconv_stack_builder(configs, inplanes; width_mult,
                                                     se_from_explanes = true,
                                                     se_round_fn = _round_channels)
    append!(layers, cnn_stages(get_layers, block_repeats, +))
    # building last layers
    explanes = _round_channels(configs[end][3] * width_mult, 8)
    midplanes = _round_channels(explanes * configs[end][4], 8)
    headplanes = _round_channels(max_width * width_mult, 8)
    append!(layers, conv_norm((1, 1), explanes, midplanes, hardswish))
    return Chain(Chain(layers...),
                 create_classifier(midplanes, headplanes, nclasses,
                                   (hardswish, identity); dropout_rate))
end

# Layer configurations for small and large models for MobileNetv3
# f: mbconv block function - we use `mbconv` for all blocks
# k: kernel size
# c: output channels
# e: expansion factor
# s: stride
# n: number of repeats
# r: squeeze and excite reduction factor
# a: activation function
# Data is organised as (f, k, c, e, s, n, r, a)
const MOBILENETV3_CONFIGS = Dict(:small => [
                                     (mbconv, 3, 16, 1, 2, 1, 4, relu),
                                     (mbconv, 3, 24, 4.5, 2, 1, nothing, relu),
                                     (mbconv, 3, 24, 3.67, 1, 1, nothing, relu),
                                     (mbconv, 5, 40, 4, 2, 1, 4, hardswish),
                                     (mbconv, 5, 40, 6, 1, 2, 4, hardswish),
                                     (mbconv, 5, 48, 3, 1, 2, 4, hardswish),
                                     (mbconv, 5, 96, 6, 1, 3, 4, hardswish),
                                 ],
                                 :large => [
                                     (mbconv, 3, 16, 1, 1, 1, nothing, relu),
                                     (mbconv, 3, 24, 4, 2, 1, nothing, relu),
                                     (mbconv, 3, 24, 3, 1, 1, nothing, relu),
                                     (mbconv, 5, 40, 3, 2, 1, 4, relu),
                                     (mbconv, 5, 40, 3, 1, 2, 4, relu),
                                     (mbconv, 3, 80, 6, 2, 1, nothing, hardswish),
                                     (mbconv, 3, 80, 2.5, 1, 1, nothing, hardswish),
                                     (mbconv, 3, 80, 2.3, 1, 2, nothing, hardswish),
                                     (mbconv, 3, 112, 6, 1, 2, 4, hardswish),
                                     (mbconv, 5, 160, 6, 1, 3, 4, hardswish),
                                 ])

"""
    MobileNetv3(config::Symbol; width_mult::Real = 1, pretrain::Bool = false,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv3 model with the specified configuration.
([reference](https://arxiv.org/abs/1905.02244)).
Set `pretrain = true` to load the model with pre-trained weights for ImageNet.

# Arguments

  - `config`: :small or :large for the size of the model (see paper).
  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `pretrain`: whether to load the pre-trained weights for ImageNet
  - `inchannels`: number of input channels
  - `nclasses`: the number of output classes

!!! warning
    
    `MobileNetv3` does not currently support pretrained weights.

See also [`mobilenetv3`](#).
"""
struct MobileNetv3
    layers::Any
end
@functor MobileNetv3

function MobileNetv3(config::Symbol; width_mult::Real = 1, pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, [:small, :large])
    max_width = config == :large ? 1280 : 1024
    layers = mobilenetv3(MOBILENETV3_CONFIGS[config]; width_mult, max_width, inchannels,
                         nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv3", config))
    end
    return MobileNetv3(layers)
end

(m::MobileNetv3)(x) = m.layers(x)

backbone(m::MobileNetv3) = m.layers[1]
classifier(m::MobileNetv3) = m.layers[2]
