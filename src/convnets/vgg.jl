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
function vgg_block(ifilters::Integer, ofilters::Integer, depth::Integer, batchnorm::Bool)
    k = (3, 3)
    p = (1, 1)
    layers = []
    for _ in 1:depth
        if batchnorm
            append!(layers, conv_norm(k, ifilters, ofilters; pad = p))
        else
            push!(layers, Conv(k, ifilters => ofilters, relu; pad = p))
        end
        ifilters = ofilters
    end
    ifilters = ofilters
    return layers
end

"""
    vgg_convolutional_layers(config, batchnorm, inchannels)

Create VGG convolution layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `config`: vector of tuples `(output_channels, num_convolutions)`
    for each block (see [`Metalhead.vgg_block`](@ref))
  - `batchnorm`: set to `true` to include batch normalization after each convolution
  - `inchannels`: number of input channels
"""
function vgg_convolutional_layers(config::AbstractVector{<:Tuple}, batchnorm::Bool,
                                  inchannels::Integer)
    layers = []
    ifilters = inchannels
    for c in config
        append!(layers, vgg_block(ifilters, c..., batchnorm))
        push!(layers, MaxPool((2, 2); stride = 2))
        ifilters, _ = c
    end
    return layers
end

"""
    vgg_classifier_layers(imsize, nclasses, fcsize, dropout_prob)

Create VGG classifier (fully connected) layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `imsize`: tuple `(width, height, channels)` indicating the size after
    the convolution layers (see [`Metalhead.vgg_convolutional_layers`](@ref))
  - `nclasses`: number of output classes
  - `fcsize`: input and output size of the intermediate fully connected layer
  - `dropout_prob`: the dropout level between each fully connected layer
"""
function vgg_classifier_layers(imsize::NTuple{3, <:Integer}, nclasses::Integer,
                               fcsize::Integer, dropout_prob)
    return Chain(MLUtils.flatten,
                 Dense(prod(imsize), fcsize, relu),
                 Dropout(dropout_prob),
                 Dense(fcsize, fcsize, relu),
                 Dropout(dropout_prob),
                 Dense(fcsize, nclasses))
end

"""
    vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout_prob)

Create a VGG model
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `imsize`: input image width and height as a tuple
  - `config`: the configuration for the convolution layers
    (see [`Metalhead.vgg_convolutional_layers`](@ref))
  - `inchannels`: number of input channels
  - `batchnorm`: set to `true` to use batch normalization after each convolution
  - `nclasses`: number of output classes
  - `fcsize`: intermediate fully connected layer size
    (see [`Metalhead.vgg_classifier_layers`](@ref))
  - `dropout_prob`: dropout level between fully connected layers
"""
function vgg(imsize::Dims{2}; config, batchnorm::Bool = false, fcsize::Integer = 4096,
             dropout_prob = 0.0, inchannels::Integer = 3, nclasses::Integer = 1000)
    conv = vgg_convolutional_layers(config, batchnorm, inchannels)
    imsize = outputsize(conv, (imsize..., inchannels); padbatch = true)[1:3]
    class = vgg_classifier_layers(imsize, nclasses, fcsize, dropout_prob)
    return Chain(Chain(conv...), class)
end

const VGG_CONV_CONFIGS = Dict(:A => [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
                              :B => [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
                              :D => [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
                              :E => [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)])

const VGG_CONFIGS = Dict(11 => :A, 13 => :B, 16 => :D, 19 => :E)

"""
    VGG(imsize::Dims{2}; config, inchannels, batchnorm = false, nclasses, fcsize, dropout_prob)

Construct a VGG model with the specified input image size. Typically, the image size is `(224, 224)`.

## Keyword Arguments:

  - `config` : VGG convolutional block configuration. It is defined as a vector of tuples
    `(output_channels, num_convolutions)` for each block
  - `inchannels`: number of input channels
  - `batchnorm`: set to `true` to use batch normalization after each convolution
  - `nclasses`: number of output classes
  - `fcsize`: intermediate fully connected layer size
    (see [`Metalhead.vgg_classifier_layers`](@ref))
  - `dropout_prob`: dropout level between fully connected layers
"""
struct VGG
    layers::Any
end
@functor VGG

function VGG(imsize::Dims{2}; config, batchnorm::Bool = false, dropout_prob = 0.5,
             inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = vgg(imsize; config, inchannels, batchnorm, nclasses, dropout_prob)
    return VGG(layers)
end

(m::VGG)(x) = m.layers(x)

backbone(m::VGG) = m.layers[1]
classifier(m::VGG) = m.layers[2]

"""
    VGG(depth::Integer; pretrain::Bool = false, batchnorm::Bool = false,
        inchannels::Integer = 3, nclasses::Integer = 1000)

Create a VGG style model with specified `depth`.
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `depth`: the depth of the VGG model. Must be one of [11, 13, 16, 19].
  - `pretrain`: set to `true` to load pre-trained model weights for ImageNet
  - `batchnorm`: set to `true` to use batch normalization after each convolution
  - `inchannels`: number of input channels
  - `nclasses`: number of output classes

See also [`vgg`](@ref).
"""
function VGG(depth::Integer; pretrain::Bool = false, batchnorm::Bool = false,
             inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(depth, keys(VGG_CONFIGS))
    model = VGG((224, 224); config = VGG_CONV_CONFIGS[VGG_CONFIGS[depth]], batchnorm,
                inchannels, nclasses)
    if pretrain
        artifact_name = string("vgg", depth)
        if batchnorm
            artifact_name *= "_bn"
        else
            artifact_name *= "-IMAGENET1K_V1"
        end
        loadpretrain!(model, artifact_name)
    end
    return model
end
