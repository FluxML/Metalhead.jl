"""
    mixerblock(planes, npatches; mlp_ratio = (0.5, 4.0), activation = gelu, dropout = 0., 
               drop_path_rate = 0.)

Creates a feedforward block for the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601))

# Arguments:
- `planes`: the number of planes in the block
- `npatches`: the number of patches of the image
- `mlp_ratio`: number(s) that determine(s) the number of hidden channels in the token mixing MLP 
               and/or the channel mixing MLP as a ratio to the number of planes in the block.
- `activation`: the activation function to use in the MLP blocks
- `dropout`: the dropout rate to use in the MLP blocks
- `drop_path_rate`: Stochastic depth rate
"""
function mixerblock(planes, npatches; mlp_ratio = (0.5, 4.0), activation = gelu, 
                    dropout = 0., drop_path_rate = 0.)
  tokenplanes, channelplanes = [Int(r * planes) for r in mlp_ratio]
  return Chain(SkipConnection(Chain(LayerNorm(planes),
                                    x -> permutedims(x, (2, 1, 3)),
                                    mlp_block(npatches, tokenplanes; activation, dropout),
                                    x -> permutedims(x, (2, 1, 3)),
                                    DropPath(drop_path_rate)), +),
               SkipConnection(Chain(LayerNorm(planes),
                                    mlp_block(planes, channelplanes; activation, dropout),
                                    DropPath(drop_path_rate)), +))
end

"""
    mlpmixer(imsize::NTuple{2} = (224, 224); inchannels = 3, patch_size::NTuple{2} = (16, 16), 
             embedplanes = 512, depth = 12, mlp_ratio::NTuple{2} = (0.5, 4.0), dropout = 0.,
             drop_path_rate = 0., nclasses = 1000)

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments
- `imsize`: the size of the input image
- `inchannels`: the number of input channels
- `patch_size`: the size of the patches
- `embedplanes`: the number of channels after the patch embedding (denotes the hidden dimension)
- `depth`: the number of blocks in the model
- `mlp_ratio`: number(s) that determine(s) the number of hidden channels in the token mixing MLP 
               and/or the channel mixing MLP as a ratio to the number of planes in the block.
- `dropout`: the dropout rate to use in the MLP blocks
- `drop_path_rate`: Stochastic depth rate
- `nclasses`: number of output classes
"""
function mlpmixer(block, imsize::NTuple{2} = (224, 224); inchannels = 3, norm_layer = LayerNorm,
                  patch_size::NTuple{2} = (16, 16), embedplanes = 512, depth = 12, 
                  mlp_ratio = (0.5, 4.0), dropout = 0., drop_path_rate = 0., 
                  nclasses = 1000)
  npatches = prod(imsize .÷ patch_size)
  dp_rates = LinRange{Float32}(0., drop_path_rate, depth)
  layers = Chain(PatchEmbedding(imsize; inchannels, patch_size, embedplanes),
                 [block(embedplanes, npatches; mlp_ratio, dropout, drop_path_rate = dp_rates[i]) 
                  for i in 1:depth]...)

  classification_head = Chain(norm_layer(embedplanes), seconddimmean, Dense(embedplanes, nclasses))
  return Chain(layers, classification_head)
end

# Configurations for MLPMixer models
mixer_configs = Dict(:small => Dict(:depth => 8,  :planes => 512),
                     :base  => Dict(:depth => 12, :planes => 768),
                     :large => Dict(:depth => 24, :planes => 1024),
                     :huge  => Dict(:depth => 32, :planes => 1280))

struct MLPMixer
  layers
end

"""
    MLPMixer(size::Symbol = :base; patch_size::Int = 16, imsize::NTuple{2} = (224, 224),
             drop_path_rate = 0., nclasses = 1000)

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments
- `size`: the size of the model - one of `small`, `base`, `large` or `huge`
- `patch_size`: the size of the patches
- `imsize`: the size of the input image
- `drop_path_rate`: Stochastic depth rate
- `nclasses`: number of output classes

See also [`Metalhead.mlpmixer`](#).
"""
function MLPMixer(size::Symbol = :base; patch_size::Int = 16, imsize::NTuple{2} = (224, 224),
                  drop_path_rate = 0., nclasses = 1000)
  @assert size in keys(mixer_configs) "`size` must be one of $(keys(mixer_configs))"
  patch_size = _to_tuple(patch_size)
  depth = mixer_configs[size][:depth]
  embedplanes = mixer_configs[size][:planes]
  layers = mlpmixer(mixerblock, imsize; patch_size, embedplanes, depth, drop_path_rate, nclasses)
  MLPMixer(layers)
end

@functor MLPMixer

(m::MLPMixer)(x) = m.layers(x)

backbone(m::MLPMixer) = m.layers[1]
classifier(m::MLPMixer) = m.layers[2]


"""
    resmixerblock(planes, npatches; dropout = 0., drop_path_rate = 0., mlp_ratio = 4.0,
                  activation = gelu, λ = 1e-4)

Creates a block for the ResMixer architecture.
([reference](https://arxiv.org/abs/2105.03404)).

# Arguments
- `planes`: the number of planes in the block
- `npatches`: the number of patches of the image
- `dropout`: the dropout rate to use in the MLP blocks
- `drop_path_rate`: Stochastic depth rate
- `mlp_ratio`: ratio of the number of hidden channels in the channel mixing MLP to the number 
               of planes in the block
- `activation`: the activation function to use in the MLP blocks
- `λ`: init 
"""
function resmixerblock(planes, npatches; dropout = 0., drop_path_rate = 0., mlp_ratio = 4.0,
                       activation = gelu, λ = 1e-4)
return Chain(SkipConnection(Chain(Flux.Diagonal(planes),
                                  x -> permutedims(x, (2, 1, 3)),
                                  Dense(npatches, npatches),
                                  x -> permutedims(x, (2, 1, 3)),
                                  LayerScale(λ, planes),
                                  DropPath(drop_path_rate)), +),
             SkipConnection(Chain(Flux.Diagonal(planes),
                                  mlp_block(planes, Int(mlp_ratio * planes); dropout, activation),
                                  LayerScale(λ, planes),
                                  DropPath(drop_path_rate)), +))
end

struct ResMLP
  layers
end

"""
    ResMLP(size::Symbol = :base; patch_size::Int = 16, imsize::NTuple{2} = (224, 224),
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
function ResMLP(size::Symbol = :base; patch_size::Int = 16, imsize::NTuple{2} = (224, 224),
                drop_path_rate = 0., nclasses = 1000)
  @assert size in keys(mixer_configs) "`size` must be one of $(keys(mixer_configs))"
  patch_size = _to_tuple(patch_size)
  depth = mixer_configs[size][:depth]
  embedplanes = mixer_configs[size][:planes]
  layers = mlpmixer(resmixerblock, imsize; mlp_ratio = 4.0, patch_size, embedplanes, 
                    depth, nclasses)
  ResMLP(layers)
end

@functor ResMLP

(m::ResMLP)(x) = m.layers(x)

backbone(m::ResMLP) = m.layers[1]
classifier(m::ResMLP) = m.layers[2]
