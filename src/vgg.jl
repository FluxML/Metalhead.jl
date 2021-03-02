using Flux: convfilter, outdims

# Build a VGG block
#  ifilters: number of input filters
#  ofilters: number of output filters
#  batchnorm: add batchnorm
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

# Build convolutionnal layers
#  config: :A (vgg11) :B (vgg13) :D (vgg16) :E (vgg19)
#  inchannels: number of channels in input image (3 for RGB)
function convolutional_layers(config, batchnorm, inchannels)
  layers = []
  ifilters = inchannels
  for c in config
    append!(layers, vgg_block(ifilters, c..., batchnorm))
    push!(layers, MaxPool((2,2)))
    ifilters, _ = c
  end
  return layers
end

# Build classification layers
#  imsize: image size (w, h, c)
#  nclasses: number of classes
#  fcsize: size of fully connected layers (usefull for smaller nclasses than ImageNet)
#  dropout: dropout obviously
function classifier_layers(imsize, nclasses, fcsize, dropout)
  layers = []
  push!(layers, flatten)
  push!(layers, Dense(Int(prod(imsize)), fcsize, relu))
  push!(layers, Dropout(dropout))
  push!(layers, Dense(fcsize, fcsize, relu))
  push!(layers, Dropout(dropout))
  push!(layers, Dense(fcsize, nclasses))

  return layers
end

function vgg(imsize; config, inchannels, batchnorm=false, nclasses, fcsize, dropout)
  conv = convolutional_layers(config, batchnorm, inchannels)
  imsize = outputsize(conv, (imsize..., inchannels); padbatch=true)[1:2]
  class = classifier_layers((imsize..., config[end][1]), nclasses, fcsize, dropout)
  return Chain(conv..., class...)
end

const configs = Dict(:A => [(64,1), (128,1), (256,2), (512,2), (512,2)],
                     :B => [(64,2), (128,2), (256,2), (512,2), (512,2)],
                     :D => [(64,2), (128,2), (256,3), (512,3), (512,3)],
                     :E => [(64,2), (128,2), (256,4), (512,4), (512,4)])

function vgg11(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:A],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg11")
  return model
end

function vgg11bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:A],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg11bn")
  return model
end

function vgg13(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:B],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg13")
  return model
end

function vgg13bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:B],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg13bn")
  return model
end

function vgg16(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:D],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg16")
  return model
end

function vgg16bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:D],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg16bn")
  return model
end

function vgg19(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:E],
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && Flux.loadparams!(model, weights("vgg19"))
  return model
end

function vgg19bn(imsize=(224, 224); inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5, pretrain=false)
  model = vgg(imsize; config=configs[:E],
                      batchnorm=true,
                      inchannels=inchannels,
                      nclasses=nclasses,
                      fcsize=fcsize,
                      dropout=dropout)

  pretrain && pretrain_error("vgg19bn")
  return model
end