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
  - `inchannels`: The number of channels in the input.
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
    head = Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten, Dense(planes, nclasses))
    return Chain(Chain(stem..., Chain(blocks)), head)
end

const CONVMIXER_CONFIGS = Dict(:base => Dict(:planes => 1536, :depth => 20,
                                             :kernel_size => (9, 9),
                                             :patch_size => (7, 7)),
                               :small => Dict(:planes => 768, :depth => 32,
                                              :kernel_size => (7, 7),
                                              :patch_size => (7, 7)),
                               :large => Dict(:planes => 1024, :depth => 20,
                                              :kernel_size => (9, 9),
                                              :patch_size => (7, 7)))

"""
    ConvMixer(mode::Symbol; inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a ConvMixer model.
([reference](https://arxiv.org/abs/2201.09792))

# Arguments

  - `mode`: the mode of the model, either `:base`, `:small` or `:large`
  - `inchannels`: The number of channels in the input.
  - `nclasses`: number of classes in the output
"""
struct ConvMixer
    layers::Any
end
@functor ConvMixer

function ConvMixer(mode::Symbol; inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(mode, keys(CONVMIXER_CONFIGS))
    planes = CONVMIXER_CONFIGS[mode][:planes]
    depth = CONVMIXER_CONFIGS[mode][:depth]
    kernel_size = CONVMIXER_CONFIGS[mode][:kernel_size]
    patch_size = CONVMIXER_CONFIGS[mode][:patch_size]
    layers = convmixer(planes, depth; inchannels, kernel_size, patch_size, nclasses)
    return ConvMixer(layers)
end

(m::ConvMixer)(x) = m.layers(x)

backbone(m::ConvMixer) = m.layers[1]
classifier(m::ConvMixer) = m.layers[2]
