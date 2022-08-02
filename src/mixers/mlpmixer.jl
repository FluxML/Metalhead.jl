"""
    mixerblock(planes::Integer, npatches::Integer; mlp_layer = mlp_block,
               mlp_ratio = (0.5, 4.0), dropout_rate = 0.0, drop_path_rate = 0.0,
               activation = gelu)

Creates a feedforward block for the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601))

# Arguments

  - `planes`: the number of planes in the block
  - `npatches`: the number of patches of the input
  - `mlp_ratio`: number(s) that determine(s) the number of hidden channels in the token mixing MLP
    and/or the channel mixing MLP as a ratio to the number of planes in the block.
  - `mlp_layer`: the MLP layer to use in the block
  - `dropout_rate`: the dropout rate to use in the MLP blocks
  - `drop_path_rate`: Stochastic depth rate
  - `activation`: the activation function to use in the MLP blocks
"""
function mixerblock(planes::Integer, npatches::Integer; mlp_layer = mlp_block,
                    mlp_ratio::NTuple{2, Number} = (0.5, 4.0), dropout_rate = 0.0,
                    drop_path_rate = 0.0, activation = gelu)
    tokenplanes, channelplanes = Int.(planes .* mlp_ratio)
    return Chain(SkipConnection(Chain(LayerNorm(planes),
                                      swapdims((2, 1, 3)),
                                      mlp_layer(npatches, tokenplanes; activation,
                                                dropout_rate),
                                      swapdims((2, 1, 3)),
                                      DropPath(drop_path_rate)), +),
                 SkipConnection(Chain(LayerNorm(planes),
                                      mlp_layer(planes, channelplanes; activation,
                                                dropout_rate),
                                      DropPath(drop_path_rate)), +))
end

"""
    MLPMixer(size::Symbol; patch_size::Dims{2} = (16, 16), imsize::Dims{2} = (224, 224),
             inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments

  - `size`: the size of the model - one of `small`, `base`, `large` or `huge`
  - `patch_size`: the size of the patches
  - `imsize`: the size of the input image
  - `drop_path_rate`: Stochastic depth rate
  - `inchannels`: the number of input channels
  - `nclasses`: number of output classes

See also [`Metalhead.mlpmixer`](#).
"""
struct MLPMixer
    layers::Any
end
@functor MLPMixer

function MLPMixer(size::Symbol; imsize::Dims{2} = (224, 224),
                  patch_size::Dims{2} = (16, 16),
                  inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(size, keys(MIXER_CONFIGS))
    depth = MIXER_CONFIGS[size][:depth]
    embedplanes = MIXER_CONFIGS[size][:planes]
    layers = mlpmixer(mixerblock, imsize; patch_size, embedplanes, depth, inchannels,
                      nclasses)
    return MLPMixer(layers)
end

(m::MLPMixer)(x) = m.layers(x)

backbone(m::MLPMixer) = m.layers[1]
classifier(m::MLPMixer) = m.layers[2]
