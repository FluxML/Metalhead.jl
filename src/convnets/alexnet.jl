"""
    alexnet(; dropout_prob = 0.5, inchannels::Integer = 3, nclasses::Integer = 1000)

Create an AlexNet model
([reference](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)).

# Arguments

  - `dropout_prob`: dropout probability for the classifier
  - `inchannels`: The number of input channels.
  - `nclasses`: the number of output classes
"""
function alexnet(; dropout_prob = 0.5, inchannels::Integer = 3, nclasses::Integer = 1000)
    backbone = Chain(Conv((11, 11), inchannels => 64, relu; stride = 4, pad = 2),
                     MaxPool((3, 3); stride = 2),
                     Conv((5, 5), 64 => 192, relu; pad = 2),
                     MaxPool((3, 3); stride = 2),
                     Conv((3, 3), 192 => 384, relu; pad = 1),
                     Conv((3, 3), 384 => 256, relu; pad = 1),
                     Conv((3, 3), 256 => 256, relu; pad = 1),
                     MaxPool((3, 3); stride = 2))
    classifier = Chain(AdaptiveMeanPool((6, 6)), MLUtils.flatten,
                       Dropout(dropout_prob),
                       Dense(256 * 6 * 6, 4096, relu),
                       Dropout(dropout_prob),
                       Dense(4096, 4096, relu),
                       Dense(4096, nclasses))
    return Chain(backbone, classifier)
end

"""
    AlexNet(; pretrain::Bool = false, inchannels::Integer = 3,
            nclasses::Integer = 1000)

Create a `AlexNet`.
([reference](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)).

# Arguments

  - `pretrain`: set to `true` to load pre-trained weights for ImageNet
  - `inchannels`: The number of input channels.
  - `nclasses`: the number of output classes

!!! warning

    `AlexNet` does not currently support pretrained weights.

See also [`alexnet`](@ref).
"""
struct AlexNet
    layers::Any
end
@functor AlexNet

function AlexNet(; pretrain::Bool = false, inchannels::Integer = 3,
                 nclasses::Integer = 1000)
    layers = alexnet(; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, "AlexNet")
    end
    return AlexNet(layers)
end

(m::AlexNet)(x) = m.layers(x)

backbone(m::AlexNet) = m.layers[1]
classifier(m::AlexNet) = m.layers[2]
