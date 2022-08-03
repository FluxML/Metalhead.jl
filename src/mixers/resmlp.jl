"""
    resmixerblock(planes, npatches; dropout_rate = 0., drop_path_rate = 0., mlp_ratio = 4.0,
                  activation = gelu, layerscale_init = 1e-4)

Creates a block for the ResMixer architecture.
([reference](https://arxiv.org/abs/2105.03404)).

# Arguments

  - `planes`: the number of planes in the block
  - `npatches`: the number of patches of the input
  - `mlp_ratio`: ratio of the number of hidden channels in the channel mixing MLP to the number
    of planes in the block
  - `mlp_layer`: the MLP block to use
  - `dropout_rate`: the dropout rate to use in the MLP blocks
  - `drop_path_rate`: Stochastic depth rate
  - `activation`: the activation function to use in the MLP blocks
  - `layerscale_init`: initialisation constant for the LayerScale
"""
function resmixerblock(planes::Integer, npatches::Integer; mlp_layer = mlp_block,
                       mlp_ratio = 4.0, layerscale_init = 1e-4, dropout_rate = 0.0,
                       drop_path_rate = 0.0, activation = gelu)
    return Chain(SkipConnection(Chain(Flux.Scale(planes),
                                      swapdims((2, 1, 3)),
                                      Dense(npatches, npatches),
                                      swapdims((2, 1, 3)),
                                      LayerScale(planes, layerscale_init),
                                      DropPath(drop_path_rate)), +),
                 SkipConnection(Chain(Flux.Scale(planes),
                                      mlp_layer(planes, floor(Int, mlp_ratio * planes);
                                                dropout_rate, activation),
                                      LayerScale(planes, layerscale_init),
                                      DropPath(drop_path_rate)), +))
end

"""
    ResMLP(config::Symbol; patch_size::Dims{2} = (16, 16), imsize::Dims{2} = (224, 224),
           inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a model with the ResMLP architecture.
([reference](https://arxiv.org/abs/2105.03404)).

# Arguments

  - `config`: the size of the model - one of `small`, `base`, `large` or `huge`
  - `patch_size`: the size of the patches
  - `imsize`: the size of the input image
  - `inchannels`: the number of input channels
  - `nclasses`: number of output classes

See also [`Metalhead.mlpmixer`](#).
"""
struct ResMLP
    layers::Any
end
@functor ResMLP

function ResMLP(config::Symbol; imsize::Dims{2} = (224, 224),
                patch_size::Dims{2} = (16, 16),
                inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(config, keys(MIXER_CONFIGS))
    layers = mlpmixer(resmixerblock, imsize; mlp_ratio = 4.0, patch_size,
                      MIXER_CONFIGS[config]..., inchannels, nclasses)
    return ResMLP(layers)
end

(m::ResMLP)(x) = m.layers(x)

backbone(m::ResMLP) = m.layers[1]
classifier(m::ResMLP) = m.layers[2]
