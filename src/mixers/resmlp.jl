"""
    resmixerblock(planes, npatches; dropout_rate = 0., drop_path_rate = 0., mlp_ratio = 4.0,
                  activation = gelu, λ = 1e-4)

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
  - `λ`: initialisation constant for the LayerScale
"""
function resmixerblock(planes, npatches; mlp_ratio = 4.0, mlp_layer = mlp_block,
                       dropout_rate = 0.0, drop_path_rate = 0.0, activation = gelu,
                       λ = 1e-4)
    return Chain(SkipConnection(Chain(Flux.Scale(planes),
                                      swapdims((2, 1, 3)),
                                      Dense(npatches, npatches),
                                      swapdims((2, 1, 3)),
                                      LayerScale(planes, λ),
                                      DropPath(drop_path_rate)), +),
                 SkipConnection(Chain(Flux.Scale(planes),
                                      mlp_layer(planes, Int(mlp_ratio * planes);
                                                dropout_rate,
                                                activation),
                                      LayerScale(planes, λ),
                                      DropPath(drop_path_rate)), +))
end

struct ResMLP
    layers::Any
end
@functor ResMLP

"""
    ResMLP(size::Symbol = :base; patch_size::Dims{2} = (16, 16), imsize::Dims{2} = (224, 224),
           drop_path_rate = 0., nclasses = 1000)

Creates a model with the ResMLP architecture.
([reference](https://arxiv.org/abs/2105.03404)).

# Arguments

  - `size`: the size of the model - one of `small`, `base`, `large` or `huge`
  - `patch_size`: the size of the patches
  - `imsize`: the size of the input image
  - `drop_path_rate`: Stochastic depth rate
  - `nclasses`: number of output classes

See also [`Metalhead.mlpmixer`](#).
"""
function ResMLP(size::Symbol = :base; patch_size::Dims{2} = (16, 16),
                imsize::Dims{2} = (224, 224), drop_path_rate = 0.0, nclasses = 1000)
    _checkconfig(size, keys(MIXER_CONFIGS))
    depth = MIXER_CONFIGS[size][:depth]
    embedplanes = MIXER_CONFIGS[size][:planes]
    layers = mlpmixer(resmixerblock, imsize; mlp_ratio = 4.0, patch_size, embedplanes,
                      drop_path_rate, depth, nclasses)
    return ResMLP(layers)
end

(m::ResMLP)(x) = m.layers(x)

backbone(m::ResMLP) = m.layers[1]
classifier(m::ResMLP) = m.layers[2]
