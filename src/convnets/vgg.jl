"""
    vgg_block(ifilters, ofilters, depth, batchnorm)

A VGG block of convolution layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `ifilters`: number of input feature maps
- `ofilters`: number of output feature maps
- `depth`: number of convolution/convolution + batch norm layers
- `batchnorm`: set to `true` to include batch normalization after each convolution
"""
function vgg_block(ifilters, ofilters, depth, batchnorm)
  k = (3,3)
  p = (1,1)
  layers = []
  for _ in 1:depth
    if batchnorm
      append!(layers, conv_bn(k, ifilters, ofilters; pad = p, bias = false))
    else
      push!(layers, Conv(k, ifilters => ofilters, relu, pad = p))
    end
    ifilters = ofilters
  end
  return layers
end

"""
    vgg_convolutional_layers(config, batchnorm, inchannels)

Create VGG convolution layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `config`: vector of tuples `(output_channels, num_convolutions)`
            for each block (see [`Metalhead.vgg_block`](#))
- `batchnorm`: set to `true` to include batch normalization after each convolution
- `inchannels`: number of input channels
"""
function vgg_convolutional_layers(config, batchnorm, inchannels)
  layers = []
  ifilters = inchannels
  for c in config
    append!(layers, vgg_block(ifilters, c..., batchnorm))
    push!(layers, MaxPool((2,2), stride=2))
    ifilters, _ = c
  end
  return layers
end

"""
    vgg_classifier_layers(imsize, nclasses, fcsize, dropout)

Create VGG classifier (fully connected) layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `imsize`: tuple `(width, height, channels)` indicating the size after
            the convolution layers (see [`Metalhead.vgg_convolutional_layers`](#))
- `nclasses`: number of output classes
- `fcsize`: input and output size of the intermediate fully connected layer
- `dropout`: the dropout level between each fully connected layer
"""
function vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
  return Chain(MLUtils.flatten,
               Dense(Int(prod(imsize)), fcsize, relu),
               Dropout(dropout),
               Dense(fcsize, fcsize, relu),
               Dropout(dropout),
               Dense(fcsize, nclasses))
end

"""
    vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)

Create a VGG model
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `imsize`: input image width and height as a tuple
- `config`: the configuration for the convolution layers
            (see [`Metalhead.vgg_convolutional_layers`](#))
- `inchannels`: number of input channels
- `batchnorm`: set to `true` to use batch normalization after each convolution
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
"""
function vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)
  conv = vgg_convolutional_layers(config, batchnorm, inchannels)
  imsize = Flux.outputsize(conv, (imsize..., inchannels); padbatch = true)[1:3]
  class = vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
  return Chain(Chain(conv), class)
end

const vgg_conv_config = Dict(:A => [(64,1), (128,1), (256,2), (512,2), (512,2)],
                             :B => [(64,2), (128,2), (256,2), (512,2), (512,2)],
                             :D => [(64,2), (128,2), (256,3), (512,3), (512,3)],
                             :E => [(64,2), (128,2), (256,4), (512,4), (512,4)])

const vgg_config = Dict(11 => :A,
                        13 => :B,
                        16 => :D,
                        19 => :E)

struct VGG
  layers
end

"""
    VGG(imsize::Dims{2}; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)

Construct a VGG model with the specified input image size. Typically, the image size is `(224, 224)`.

## Keyword Arguments:
- `config` : VGG convolutional block configuration. It is defined as a vector of tuples `(output_channels, num_convolutions)` for each block 
- `inchannels`::Int : number of input channels
- `batchnorm`::Bool : set to `true` to use batch normalization after each convolution
- `nclasses`::Int : number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
"""
function VGG(imsize::Dims{2};
             config, inchannels, batchnorm = false, nclasses, fcsize, dropout)
  layers = vgg(imsize; config = config,
                       inchannels = inchannels,
                       batchnorm = batchnorm,
                       nclasses = nclasses,
                       fcsize = fcsize,
                       dropout = dropout)

  VGG(layers)
end

@functor VGG

(m::VGG)(x) = m.layers(x)

backbone(m::VGG) = m.layers[1]
classifier(m::VGG) = m.layers[2]

"""
    VGG(depth::Int = 16; pretrain = false, batchnorm = false)

Create a VGG style model with specified `depth`. Available values include (11, 13, 16, 19).
([reference](https://arxiv.org/abs/1409.1556v6)).
See also [`VGG`](#).

!!! warning
    `VGG` does not currently support pretrained weights.

# Arguments
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function VGG(depth::Int = 16; pretrain = false, batchnorm = false, nclasses = 1000)
  @assert depth in keys(vgg_config) "depth must be from one in $(sort(collect(keys(vgg_config))))"

  model = VGG((224, 224); config = vgg_conv_config[vgg_config[depth]],
                          inchannels = 3,
                          batchnorm = batchnorm,
                          nclasses = nclasses,
                          fcsize = 4096,
                          dropout = 0.5)

  if pretrain && !batchnorm
    loadpretrain!(model, string("VGG", depth))
  elseif pretrain
    loadpretrain!(model, "VGG$(depth)-BN)")
  end
  model
end

# deprecations
@deprecate VGG11(; kw...) VGG(11; kw...)
@deprecate VGG13(; kw...) VGG(13; kw...)
@deprecate VGG16(; kw...) VGG(16; kw...)
@deprecate VGG19(; kw...) VGG(19; kw...)
