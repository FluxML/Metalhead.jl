"""
    mobilenetv1(width_mult::Real, config::AbstractVector{<:Tuple}; 
                activation = relu, dropout_rate = nothing,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv1 model ([reference](https://arxiv.org/abs/1704.04861v1)).

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper)

  - `configs`: A "list of tuples" configuration for each layer that details:
    
      + `dw`: Set true to use a depthwise separable convolution or false for regular convolution
      + `o`: The number of output feature maps
      + `s`: The stride of the convolutional kernel
      + `r`: The number of time this configuration block is repeated
  - `activate`: The activation function to use throughout the network
  - `dropout_rate`: The dropout rate to use in the classifier head. Set to `nothing` to disable.
  - `inchannels`: The number of input channels. The default value is 3.
  - `nclasses`: The number of output classes
"""
function mobilenetv1(config::AbstractVector{<:Tuple}; width_mult::Real = 1,
                     activation = relu, dropout_rate = nothing,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = []
    # stem of the model
    append!(layers,
            conv_norm((3, 3), inchannels, config[1][3], activation; stride = 2, pad = 1))
    # building inverted residual blocks
    get_layers, block_repeats = mbconv_stack_builder(config, config[1][3]; width_mult)
    append!(layers, cnn_stages(get_layers, block_repeats))
    return Chain(Chain(layers...),
                 create_classifier(config[end][3], nclasses; dropout_rate))
end

# Layer configurations for MobileNetv1
# f: block function - we use `dwsep_conv_bn` for all blocks
# k: kernel size
# c: output channels
# s: stride
# n: number of repeats
# a: activation function
const MOBILENETV1_CONFIGS = [
    # f, k, c, s, n, a
    (dwsep_conv_bn, 3, 64, 1, 1, relu6),
    (dwsep_conv_bn, 3, 128, 2, 1, relu6),
    (dwsep_conv_bn, 3, 128, 1, 1, relu6),
    (dwsep_conv_bn, 3, 256, 2, 1, relu6),
    (dwsep_conv_bn, 3, 256, 1, 1, relu6),
    (dwsep_conv_bn, 3, 512, 2, 1, relu6),
    (dwsep_conv_bn, 3, 512, 1, 5, relu6),
    (dwsep_conv_bn, 3, 1024, 2, 1, relu6),
    (dwsep_conv_bn, 3, 1024, 1, 1, relu6),
]

"""
    MobileNetv1(width_mult::Real = 1; pretrain::Bool = false,
                inchannels::Integer = 3, nclasses::Integer = 1000)

Create a MobileNetv1 model with the baseline configuration
([reference](https://arxiv.org/abs/1704.04861v1)).

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: The number of output classes

!!! warning
    
    `MobileNetv1` does not currently support pretrained weights.

See also [`mobilenetv1`](#).
"""
struct MobileNetv1
    layers::Any
end
@functor MobileNetv1

function MobileNetv1(width_mult::Real = 1; pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = mobilenetv1(MOBILENETV1_CONFIGS; width_mult, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv1"))
    end
    return MobileNetv1(layers)
end

(m::MobileNetv1)(x) = m.layers(x)

backbone(m::MobileNetv1) = m.layers[1]
classifier(m::MobileNetv1) = m.layers[2]
