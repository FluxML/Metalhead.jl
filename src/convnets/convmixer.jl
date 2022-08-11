"""
    convmixer(planes::Integer, depth::Integer; kernel_size = (9, 9),
              patch_size::Dims{2} = (7, 7), activation = gelu,
              inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a ConvMixer model.
([reference](https://arxiv.org/abs/2201.09792))

# Arguments

  - `planes`: number of planes in the output of each block
  - `depth`: number of layers
  - `kernel_size`: kernel size of the convolutional layers
  - `patch_size`: size of the patches
  - `activation`: activation function used after the convolutional layers
  - `inchannels`: number of input channels
  - `nclasses`: number of classes in the output
"""
function convmixer(planes::Integer, depth::Integer; kernel_size = (9, 9),
                   patch_size::Dims{2} = (7, 7), activation = gelu,
                   inchannels::Integer = 3, nclasses::Integer = 1000)
    stem = conv_norm(patch_size, inchannels, planes, activation; preact = true,
                     stride = patch_size[1])
    blocks = [Chain(SkipConnection(Chain(conv_norm(kernel_size, planes, planes, activation;
                                                   preact = true, groups = planes,
                                                   pad = SamePad())), +),
                    conv_norm((1, 1), planes, planes, activation; preact = true)...)
              for _ in 1:depth]
    return Chain(Chain(stem..., Chain(blocks...)), create_classifier(planes, nclasses))
end

const CONVMIXER_CONFIGS = Dict(:base => ((1536, 20),
                                         (kernel_size = (9, 9),
                                          patch_size = (7, 7))),
                               :small => ((768, 32),
                                          (kernel_size = (7, 7),
                                           patch_size = (7, 7))),
                               :large => ((1024, 20),
                                          (kernel_size = (9, 9),
                                           patch_size = (7, 7))))

"""
    ConvMixer(config::Symbol; inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a ConvMixer model.
([reference](https://arxiv.org/abs/2201.09792))

# Arguments

  - `config`: the size of the model, either `:base`, `:small` or `:large`
  - `inchannels`: number of input channels
  - `nclasses`: number of classes in the output
"""
struct ConvMixer
    layers::Any
end
@functor ConvMixer

function ConvMixer(config::Symbol; inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, keys(CONVMIXER_CONFIGS))
    layers = convmixer(CONVMIXER_CONFIGS[config][1]...; CONVMIXER_CONFIGS[config][2]...,
                       inchannels, nclasses)
    return ConvMixer(layers)
end

(m::ConvMixer)(x) = m.layers(x)

backbone(m::ConvMixer) = m.layers[1]
classifier(m::ConvMixer) = m.layers[2]
