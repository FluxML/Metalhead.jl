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
  push!(layers, softmax)
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

vgg11(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:A], inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg11bn(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:A], batchnorm=true, inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg13(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:B], inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg13bn(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:B], batchnorm=true, inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg16(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:D], inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg16bn(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:D], batchnorm=true, inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg19(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:E], inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)

vgg19bn(imsize; inchannels=3, nclasses=1000, fcsize=4096, dropout=0.5) =
  vgg(imsize, config=configs[:E], batchnorm=true, inchannels=inchannels, nclasses=nclasses, fcsize=fcsize, dropout=dropout)