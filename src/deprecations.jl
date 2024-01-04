# Deprecated; to be removed in a future release
function VGG(imsize::Dims{2}; config, batchnorm::Bool = false, dropout_prob = 0.5,
    inchannels::Integer = 3, nclasses::Integer = 1000)
    Base.depwarn("The `VGG(imsize; config, inchannels, batchnorm, nclasses)` constructor 
                will be deprecated in a future release. Please use `vgg(imsize; config, 
                inchannels, batchnorm, nclasses)` instead for the same functionality.", :VGG)
    layers = vgg(imsize; config, inchannels, batchnorm, nclasses, dropout_prob)
    return VGG(layers)
end