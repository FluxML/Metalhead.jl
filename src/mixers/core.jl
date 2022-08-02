"""
    mlpmixer(block, imsize::Dims{2} = (224, 224); inchannels::Integer = 3, norm_layer = LayerNorm,
             patch_size::Dims{2} = (16, 16), embedplanes = 512, drop_path_rate = 0.,
             depth = 12, nclasses::Integer = 1000, kwargs...)

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments

  - `block`: the type of mixer block to use in the model - architecture dependent
    (a constructor of the form `block(embedplanes, npatches; drop_path_rate, kwargs...)`)
  - `imsize`: the size of the input image
  - `inchannels`: the number of input channels
  - `norm_layer`: the normalization layer to use in the model
  - `patch_size`: the size of the patches
  - `embedplanes`: the number of channels after the patch embedding (denotes the hidden dimension)
  - `drop_path_rate`: Stochastic depth rate
  - `depth`: the number of blocks in the model
  - `nclasses`: number of output classes
  - `kwargs`: additional arguments (if any) to pass to the mixer block. Will use the defaults if
    not specified.
"""
function mlpmixer(block, imsize::Dims{2} = (224, 224); norm_layer = LayerNorm,
                  patch_size::Dims{2} = (16, 16), embedplanes = 512, drop_path_rate = 0.0,
                  depth = 12, inchannels::Integer = 3, nclasses::Integer = 1000, kwargs...)
    npatches = prod(imsize .÷ patch_size)
    dp_rates = linear_scheduler(drop_path_rate; depth)
    layers = Chain(PatchEmbedding(imsize; inchannels, patch_size, embedplanes),
                   Chain([block(embedplanes, npatches; drop_path_rate = dp_rates[i],
                                kwargs...)
                          for i in 1:depth]...))
    classifier = Chain(norm_layer(embedplanes), seconddimmean, Dense(embedplanes, nclasses))
    return Chain(layers, classifier)
end

# Configurations for MLPMixer models
const MIXER_CONFIGS = Dict(:small => Dict(:depth => 8, :planes => 512),
                           :base => Dict(:depth => 12, :planes => 768),
                           :large => Dict(:depth => 24, :planes => 1024),
                           :huge => Dict(:depth => 32, :planes => 1280))
