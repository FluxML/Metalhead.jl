"""
    mobilenetv1(width_mult::Real, config::AbstractVector{<:Tuple}; activation = relu,
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
  - `inchannels`: The number of input channels. The default value is 3.
  - `nclasses`: The number of output classes
"""
function mobilenetv1(width_mult::Real, config::AbstractVector{<:Tuple}; activation = relu,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = []
    for (dw, outchannels, stride, nrepeats) in config
        outchannels = floor(Int, outchannels * width_mult)
        for _ in 1:nrepeats
            layer = dw ?
                    dwsep_conv_bn((3, 3), inchannels, outchannels, activation;
                                  stride, pad = 1) :
                    conv_norm((3, 3), inchannels, outchannels, activation; stride, pad = 1)
            append!(layers, layer)
            inchannels = outchannels
        end
    end
    return Chain(Chain(layers...), create_classifier(inchannels, nclasses))
end

# Layer configurations for MobileNetv1
const MOBILENETV1_CONFIGS = [
    # dw, c, s, r
    (false, 32, 2, 1),
    (true, 64, 1, 1),
    (true, 128, 2, 1),
    (true, 128, 1, 1),
    (true, 256, 2, 1),
    (true, 256, 1, 1),
    (true, 512, 2, 1),
    (true, 512, 1, 5),
    (true, 1024, 2, 1),
    (true, 1024, 1, 1),
]

"""
    MobileNetv1(width_mult = 1; inchannels::Integer = 3, pretrain::Bool = false,
                nclasses::Integer = 1000)

Create a MobileNetv1 model with the baseline configuration
([reference](https://arxiv.org/abs/1704.04861v1)).
Set `pretrain` to `true` to load the pretrained weights for ImageNet.

# Arguments

  - `width_mult`: Controls the number of output feature maps in each block
    (with 1 being the default in the paper;
    this is usually a value between 0.1 and 1.4)
  - `inchannels`: The number of input channels.
  - `pretrain`: Whether to load the pre-trained weights for ImageNet
  - `nclasses`: The number of output classes

See also [`Metalhead.mobilenetv1`](#).
"""
struct MobileNetv1
    layers::Any
end
@functor MobileNetv1

function MobileNetv1(width_mult::Real = 1; pretrain::Bool = false,
                     inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = mobilenetv1(width_mult, MOBILENETV1_CONFIGS; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("MobileNetv1"))
    end
    return MobileNetv1(layers)
end

(m::MobileNetv1)(x) = m.layers(x)

backbone(m::MobileNetv1) = m.layers[1]
classifier(m::MobileNetv1) = m.layers[2]
