"""
    xception_block(inchannels::Integer, outchannels::Integer, nrepeats::Integer;
                   stride::Integer = 1, start_with_relu::Bool = true,
                   grow_at_start::Bool = true)

Create an Xception block.
([reference](https://arxiv.org/abs/1610.02357))

# Arguments

  - `inchannels`: number of input channels
  - `outchannels`: number of output channels.
  - `nrepeats`: number of repeats of depthwise separable convolution layers.
  - `stride`: stride by which to downsample the input.
  - `start_with_relu`: if true, start the block with a ReLU activation.
  - `grow_at_start`: if true, increase the number of channels at the first convolution.
"""
function xception_block(inchannels::Integer, outchannels::Integer, nrepeats::Integer;
                        stride::Integer = 1, start_with_relu::Bool = true,
                        grow_at_start::Bool = true)
    if outchannels != inchannels || stride != 1
        skip = conv_norm((1, 1), inchannels, outchannels, identity; stride = stride)
    else
        skip = [identity]
    end
    layers = []
    for i in 1:nrepeats
        if grow_at_start
            inc = i == 1 ? inchannels : outchannels
            outc = outchannels
        else
            inc = inchannels
            outc = i == nrepeats ? outchannels : inchannels
        end
        push!(layers, relu)
        append!(layers,
                dwsep_conv_norm((3, 3), inc, outc; pad = 1, use_norm = (false, false)))
        push!(layers, BatchNorm(outc))
    end
    layers = start_with_relu ? layers : layers[2:end]
    push!(layers, MaxPool((3, 3); stride = stride, pad = 1))
    return Parallel(+, Chain(skip...), Chain(layers...))
end

"""
    xception(; dropout_prob = nothing, inchannels::Integer = 3, nclasses::Integer = 1000)

Creates an Xception model.
([reference](https://arxiv.org/abs/1610.02357))

# Arguments

  - `dropout_prob`: probability of dropout in classifier head. Set to `nothing` to disable dropout.
  - `inchannels`: number of input channels.
  - `nclasses`: the number of output classes.
"""
function xception(; dropout_prob = 0.0, inchannels::Integer = 3, nclasses::Integer = 1000)
    backbone = Chain(conv_norm((3, 3), inchannels, 32; stride = 2)...,
                     conv_norm((3, 3), 32, 64)...,
                     xception_block(64, 128, 2; stride = 2, start_with_relu = false),
                     xception_block(128, 256, 2; stride = 2),
                     xception_block(256, 728, 2; stride = 2),
                     [xception_block(728, 728, 3) for _ in 1:8]...,
                     xception_block(728, 1024, 2; stride = 2, grow_at_start = false),
                     dwsep_conv_norm((3, 3), 1024, 1536; pad = 1)...,
                     dwsep_conv_norm((3, 3), 1536, 2048; pad = 1)...)
    return Chain(backbone, create_classifier(2048, nclasses; dropout_prob))
end

"""
    Xception(; pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000)

Creates an Xception model.
([reference](https://arxiv.org/abs/1610.02357))

# Arguments

  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet.
  - `inchannels`: number of input channels.
  - `nclasses`: the number of output classes.

!!! warning
    
    `Xception` does not currently support pretrained weights.
"""
struct Xception
    layers::Any
end
@functor Xception

function Xception(; pretrain::Bool = false, inchannels::Integer = 3,
                  nclasses::Integer = 1000)
    layers = xception(; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "xception")
    end
    return Xception(layers)
end

(m::Xception)(x) = m.layers(x)

backbone(m::Xception) = m.layers[1]
classifier(m::Xception) = m.layers[2]
