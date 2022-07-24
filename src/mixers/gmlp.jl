"""
    SpatialGatingUnit(norm, proj)

Creates a spatial gating unit as described in the gMLP paper.
([reference](https://arxiv.org/abs/2105.08050))

# Arguments

  - `norm`: the normalisation layer to use
  - `proj`: the projection layer to use
"""
struct SpatialGatingUnit{T, F}
    norm::T
    proj::F
end
@functor SpatialGatingUnit

"""
    SpatialGatingUnit(planes::Integer, npatches::Integer; norm_layer = LayerNorm)

Creates a spatial gating unit as described in the gMLP paper.
([reference](https://arxiv.org/abs/2105.08050))

# Arguments

  - `planes`: the number of planes in the block
  - `npatches`: the number of patches of the input
  - `norm_layer`: the normalisation layer to use
"""
function SpatialGatingUnit(planes::Integer, npatches::Integer; norm_layer = LayerNorm)
    gateplanes = planes ÷ 2
    norm = norm_layer(gateplanes)
    proj = Dense(2 * eps(Float32) .* rand(Float32, npatches, npatches), ones(npatches))
    return SpatialGatingUnit(norm, proj)
end

function (m::SpatialGatingUnit)(x)
    u, v = chunk(x, 2; dims = 1)
    v = m.norm(v)
    v = m.proj(permutedims(v, (2, 1, 3)))
    return u .* permutedims(v, (2, 1, 3))
end

"""
    spatial_gating_block(planes, npatches; mlp_ratio = 4.0, mlp_layer = gated_mlp_block,
                         norm_layer = LayerNorm, dropout_rate = 0.0, drop_path_rate = 0.0,
                         activation = gelu)

Creates a feedforward block based on the gMLP model architecture described in the paper.
([reference](https://arxiv.org/abs/2105.08050))

# Arguments

  - `planes`: the number of planes in the block
  - `npatches`: the number of patches of the input
  - `mlp_ratio`: ratio of the number of hidden channels in the channel mixing MLP to the number
    of planes in the block
  - `norm_layer`: the normalisation layer to use
  - `dropout_rate`: the dropout rate to use in the MLP blocks
  - `drop_path_rate`: Stochastic depth rate
  - `activation`: the activation function to use in the MLP blocks
"""
function spatial_gating_block(planes, npatches; mlp_ratio = 4.0, norm_layer = LayerNorm,
                              mlp_layer = gated_mlp_block, dropout_rate = 0.0,
                              drop_path_rate = 0.0,
                              activation = gelu)
    channelplanes = Int(mlp_ratio * planes)
    sgu = inplanes -> SpatialGatingUnit(inplanes, npatches; norm_layer)
    return SkipConnection(Chain(norm_layer(planes),
                                mlp_layer(sgu, planes, channelplanes; activation,
                                          dropout_rate),
                                DropPath(drop_path_rate)), +)
end

struct gMLP
    layers::Any
end
@functor gMLP

"""
    gMLP(size::Symbol = :base; patch_size::Dims{2} = (16, 16),
         imsize::Dims{2} = (224, 224), drop_path_rate = 0., nclasses = 1000)

Creates a model with the gMLP architecture.
([reference](https://arxiv.org/abs/2105.08050)).

# Arguments

  - `size`: the size of the model - one of `small`, `base`, `large` or `huge`
  - `patch_size`: the size of the patches
  - `imsize`: the size of the input image
  - `drop_path_rate`: Stochastic depth rate
  - `nclasses`: number of output classes

See also [`Metalhead.mlpmixer`](#).
"""
function gMLP(size::Symbol = :base; patch_size::Dims{2} = (16, 16),
              imsize::Dims{2} = (224, 224), drop_path_rate = 0.0, nclasses = 1000)
    _checkconfig(size, keys(mixer_configs))
    depth = mixer_configs[size][:depth]
    embedplanes = mixer_configs[size][:planes]
    layers = mlpmixer(spatial_gating_block, imsize; mlp_layer = gated_mlp_block,
                      patch_size, embedplanes, drop_path_rate, depth, nclasses)
    return gMLP(layers)
end

(m::gMLP)(x) = m.layers(x)

backbone(m::gMLP) = m.layers[1]
classifier(m::gMLP) = m.layers[2]
