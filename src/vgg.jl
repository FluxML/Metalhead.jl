using Flux: convfilter, outdims

"""
    vgg_block(ifilters, ofilters, depth, batchnorm)

A VGG block of convolution layers (ref: https://arxiv.org/abs/1409.1556v6).

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
  for l in 1:depth
    if batchnorm
      append!(layers, conv_bn(k, ifilters, ofilters; pad=p, usebias=false))
    else
      push!(layers, Conv(k, ifilters=>ofilters, relu, pad=p))
    end
    ifilters = ofilters
  end
  return layers
end

"""
    vgg_convolutional_layers(config, batchnorm, inchannels)

Create VGG convolution layers (ref: https://arxiv.org/abs/1409.1556v6).

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
    push!(layers, MaxPool((2,2)))
    ifilters, _ = c
  end
  return layers
end

"""
    vgg_classifier_layers(imsize, nclasses, fcsize, dropout)

Create VGG classifier (fully connected) layers (ref: https://arxiv.org/abs/1409.1556v6).

# Arguments
- `imsize`: tuple `(width, height, channels)` indicating the size after
            the convolution layers (see [`Metalhead.vgg_convolutional_layers`](#))
- `nclasses`: number of output classes
- `fcsize`: input and output size of the intermediate fully connected layer
- `dropout`: the dropout level between each fully connected layer
"""
function vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
  layers = []
  push!(layers, flatten)
  push!(layers, Dense(Int(prod(imsize)), fcsize, relu))
  push!(layers, Dropout(dropout))
  push!(layers, Dense(fcsize, fcsize, relu))
  push!(layers, Dropout(dropout))
  push!(layers, Dense(fcsize, nclasses))

  return layers
end

"""
    vgg(imsize; config, inchannels, batchnorm=false, nclasses, fcsize, dropout)

Create a VGG model (ref: https://arxiv.org/abs/1409.1556v6).

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
function vgg(imsize; config, inchannels, batchnorm=false, nclasses, fcsize, dropout)
  conv = vgg_convolutional_layers(config, batchnorm, inchannels)
  imsize = outputsize(conv, (imsize..., inchannels); padbatch=true)[1:3]
  class = vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
  return Chain(conv..., class...)
end

const vgg_config = Dict(:A => [(64,1), (128,1), (256,2), (512,2), (512,2)],
                        :B => [(64,2), (128,2), (256,2), (512,2), (512,2)],
                        :D => [(64,2), (128,2), (256,3), (512,3), (512,3)],
                        :E => [(64,2), (128,2), (256,4), (512,4), (512,4)])

"""
    vgg11(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-11 style model (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg11` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg11(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:A],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg11")
  return model
end

"""
    vgg11bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-11 style model with batch normalization (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg11bn` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg11bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:A],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg11bn")
  return model
end

"""
    vgg13(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-13 style model (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg13` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg13(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:B],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg13")
  return model
end

"""
    vgg13bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-13 style model with batch normalization (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg13bn` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg13bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:B],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg13bn")
  return model
end

"""
    vgg16(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-16 style model (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg16` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg16(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:D],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg16")
  return model
end

"""
    vgg16bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-16 style model with batch normalization (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg16bn` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg16bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:D],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg16bn")
  return model
end

"""
    vgg19(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-19 style model (ref: https://arxiv.org/abs/1409.1556v6).

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg19(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:E],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && Flux.loadparams!(model, weights("vgg19"))
  return model
end

"""
    vgg19bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)

Create a VGG-19 style model with batch normalization (ref: https://arxiv.org/abs/1409.1556v6).

!!! warning
    `vgg19bn` does not currently support pretrained weights.

# Arguments
- `imsize`: input image size as `(width, height)`
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function vgg19bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=vgg_config[:E],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg19bn")
  return model
end