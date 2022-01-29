# Utility function for creating a residual block with LayerNorm before the residual connection
residualprenorm(planes, fn) = SkipConnection(Chain(fn, LayerNorm(planes)), +)

# Utility function for 1D convolution
conv1d(inplanes, outplanes, activation) = Conv((1, ), inplanes => outplanes, activation)

"""
    feedforward(planes, expansion_factor = 4, dropout = 0., dense = Dense)

Feedforward block in the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments
  `planes`: Number of dimensions in the input and output.
  `expansion_factor`: Determines the number of dimensions in the intermediate layer.
  `activation`: Activation function to use.
  `dropout`: Dropout rate.
  `dense`: Type of dense layer to use in the feedforward block.
"""
function feedforward(planes, expansion_factor = 4, dropout = 0., dense = Dense)
  Chain(dense(planes, planes * expansion_factor, gelu),
        Dropout(dropout),
        dense(planes * expansion_factor, planes, gelu),
        Dropout(dropout))
end

struct MLPMixer
  channels
  planes
  patch_size
  num_patches
  token_mix
  channel_mix
  layers
  nclasses
end

"""
    MLPMixer(; image_size = 256, channels = 3, patch_size = 16, planes = 512, 
               depth = 12, expansion_factor = 4, dropout = 0., nclasses = 1000)

Creates a model with the MLPMixer architecture.
([reference](https://arxiv.org/pdf/2105.01601)).

# Arguments
- `image_size`: Size of the input image.
- `channels`: Number of channels in the input image.
- `patch_size`: Size of each patch fed into the network.
- `planes`: Number of dimensions in every layer after the patch expansion layer.
- `depth`: Number of layers in the network.
- `expansion_factor`: Determines the number of dimensions in the intermediate layers.
- `dropout`: Dropout rate in the feedforward blocks.
- `nclasses`: Number of classes in the output.
"""
function MLPMixer(; image_size = 256, channels = 3, patch_size = 16, planes = 512, 
                    depth = 12, expansion_factor = 4, dropout = 0., nclasses = 1000)
  @assert (image_size % patch_size) == 0 "image size must be divisible by patch size"
  
  num_patches = (image_size ÷ patch_size) ^ 2
  token_mix = conv1d
  channel_mix = Dense

  layers = [Chain(residualprenorm(planes, feedforward(num_patches, expansion_factor, 
                                  dropout, token_mix)),
                  residualprenorm(planes, feedforward(planes, expansion_factor, dropout, 
                                  channel_mix)),) for _ in 1:depth]

  MLPMixer(channels,
           planes,
           patch_size,
           num_patches,
           token_mix,
           channel_mix,
           layers,
           nclasses)
end

function (m::MLPMixer)(x)
  p = m.patch_size
  @cast x[(h2, w2, c), (h, w), b] := x[(h, h2), (w, w2), c, b]  h2 in 1:p, w2 in 1:p
  x = Dense((m.patch_size ^ 2) * m.channels, m.planes)(x)
  x = Chain(LayerNorm(m.planes), m.layers...)(x)
  @reduce x[b, c] := mean(n) x[b, n, c]
  x = Dense(m.planes, m.nclasses)(x)
end

@functor MLPMixer