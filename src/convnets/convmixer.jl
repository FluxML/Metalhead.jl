"""
    convmixer(planes, depth; inchannels = 3, kernel_size = (9, 9), patch_size::Dims{2} = 7,
              activation = gelu, nclasses = 1000)

Creates a ConvMixer model.
([reference](https://arxiv.org/abs/2201.09792))

# Arguments
- `planes`: number of planes in the output of each block
- `depth`: number of layers
- `inchannels`: number of channels in the input
- `kernel_size`: kernel size of the convolutional layers
- `patch_size`: size of the patches
- `activation`: activation function used after the convolutional layers
- `nclasses`: number of classes in the output
"""
function convmixer(planes, depth; inchannels = 3, kernel_size = (9, 9),
                   patch_size::Dims{2} = (7, 7), activation = gelu, nclasses = 1000)
    stem = conv_bn(patch_size, inchannels, planes, activation; preact = true,
                   stride = patch_size[1])
    blocks = [Chain(SkipConnection(Chain(conv_bn(kernel_size, planes, planes, activation;
                                                 preact = true, groups = planes,
                                                 pad = SamePad())), +),
                    conv_bn((1, 1), planes, planes, activation; preact = true)...)
              for _ in 1:depth]
    head = Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten, Dense(planes, nclasses))
    return Chain(Chain(stem..., Chain(blocks)), head)
end

convmixer_config = Dict(:base => Dict(:planes => 1536, :depth => 20, :kernel_size => (9, 9),
                                      :patch_size => (7, 7)),
                        :small => Dict(:planes => 768, :depth => 32, :kernel_size => (7, 7),
                                       :patch_size => (7, 7)),
                        :large => Dict(:planes => 1024, :depth => 20,
                                       :kernel_size => (9, 9),
                                       :patch_size => (7, 7)))

"""
    ConvMixer(mode::Symbol = :base; inchannels = 3, activation = gelu, nclasses = 1000)

Creates a ConvMixer model.
([reference](https://arxiv.org/abs/2201.09792))

# Arguments
- `mode`: the mode of the model, either `:base`, `:small` or `:large`
- `inchannels`: number of channels in the input
- `activation`: activation function used after the convolutional layers
- `nclasses`: number of classes in the output
"""
struct ConvMixer
    layers::Any
end

function ConvMixer(mode::Symbol = :base; inchannels = 3, activation = gelu, nclasses = 1000)
    planes = convmixer_config[mode][:planes]
    depth = convmixer_config[mode][:depth]
    kernel_size = convmixer_config[mode][:kernel_size]
    patch_size = convmixer_config[mode][:patch_size]
    layers = convmixer(planes, depth; inchannels, kernel_size, patch_size, activation,
                       nclasses)
    return ConvMixer(layers)
end

@functor ConvMixer

(m::ConvMixer)(x) = m.layers(x)

backbone(m::ConvMixer) = m.layers[1]
classifier(m::ConvMixer) = m.layers[2]
