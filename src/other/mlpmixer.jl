# Utility function for creating a residual block with LayerNorm before the residual connection
_residualprenorm(planes, fn) = SkipConnection(Chain(fn, LayerNorm(planes)), +)

# Utility function for 1D convolution
_conv1d(inplanes, outplanes, activation = identity) = Conv((1, ), inplanes => outplanes, activation)

"""
    mlpmixer(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = 16, planes = 512, 
             depth = 12, expansion_factor = 4, dropout = 0., nclasses = 1000, token_mix = 
             _conv1d, channel_mix = Dense))

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments
- `imsize`: the size of the input image
- `inchannels`: the number of input channels
- `patch_size`: the size of the patches
- `planes`: the number of channels fed into the main model
- `depth`: the number of blocks in the main model
- `expansion_factor`: the number of channels in each block
- `dropout`: the dropout rate
- `nclasses`: the number of classes in the output
- `token_mix`: the function to use for the token mixing layer
- `channel_mix`: the function to use for the channel mixing layer
"""
function mlpmixer(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = 16, planes = 512, 
                  depth = 12, expansion_factor = 4, dropout = 0., nclasses = 1000, token_mix = 
                  _conv1d, channel_mix = Dense)
                    
  im_height, im_width = imsize

  @assert (im_height % patch_size) == 0 && (im_width % patch_size == 0)
  "image size must be divisible by patch size"
  
  num_patches = (im_height ÷ patch_size) * (im_width ÷ patch_size)

  layers = []
  push!(layers, PatchEmbedding(patch_size))
  push!(layers, Dense((patch_size ^ 2) * inchannels, planes))
  append!(layers, [Chain(_residualprenorm(planes, mlp_block(num_patches, 
                                          expansion_factor * num_patches; 
                                          dropout, dense = token_mix)),
                         _residualprenorm(planes, mlp_block(planes, 
                                          expansion_factor * planes; dropout, 
                                          dense = channel_mix)),) for _ in 1:depth])

  classification_head = Chain(_seconddimmean, Dense(planes, nclasses))

  return Chain(Chain(layers...), classification_head)
end

struct MLPMixer
  layers
end

"""
    MLPMixer(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = 16, planes = 512, 
             depth = 12, expansion_factor = 4, dropout = 0., nclasses = 1000)

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments
- `imsize`: the size of the input image
- `inchannels`: the number of input channels
- `patch_size`: the size of the patches
- `planes`: the number of channels fed into the main model
- `depth`: the number of blocks in the main model
- `expansion_factor`: the number of channels in each block
- `dropout`: the dropout rate
- `nclasses`: the number of classes in the output

See also [`Metalhead.mlpmixer`](#).
"""
function MLPMixer(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = 16, planes = 512, 
                  depth = 12, expansion_factor = 4, dropout = 0., nclasses = 1000)
                    
  layers = mlpmixer(imsize; inchannels, patch_size, planes, depth, expansion_factor, dropout, 
                    nclasses)
  MLPMixer(layers)
end

@functor MLPMixer

(m::MLPMixer)(x) = m.layers(x)

backbone(m::MLPMixer) = m.layers[1]
classifier(m::MLPMixer) = m.layers[2]
