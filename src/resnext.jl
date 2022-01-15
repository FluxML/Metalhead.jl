using Flux
using Functors
using Flux: Zygote
using Flux: _big_show
using Metalhead
using Test
using Metalhead: conv_bn
using Metalhead: skip_identity
using Metalhead: skip_projection

resnextblock(inplanes, outplanes, cardinality, downsample = false) = downsample ? 
    Chain(
        conv_bn((1, 1), inplanes, outplanes[1]; stride = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 2, pad = 1, 
                    groups = cardinality, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...) :
    Chain(
        conv_bn((1, 1), inplanes, outplanes[1]; stride = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 1, pad = 1,
                    groups = cardinality, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...)


function resnext(connection = (x, y) -> @. relu(x) + relu(y); channel_config, block_config, cardinality, nclasses = 1000)
    inplanes = 128
    baseplanes = 128
    layers = []
    append!(layers, conv_bn((7, 7), 3, inplanes; stride = 2, pad = (3, 3)))
    push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))
    for (i, nrepeats) in enumerate(block_config)
        # output planes within a block
        outplanes = baseplanes .* channel_config
        # push first skip connection on using first residual
        # downsample the residual path if this is the first repetition of a block
        push!(layers, Parallel(connection, resnextblock(inplanes, outplanes, cardinality, i != 1),
                                            skip_projection(inplanes, outplanes[end], i != 1)))
        # push remaining skip connections on using second residual
        inplanes = outplanes[end]
        for _ in 2:nrepeats
            push!(layers, Parallel(connection, resnextblock(inplanes, outplanes, cardinality, false),
                                                skip_identity(inplanes, outplanes[end], false)))
            inplanes = outplanes[end]
        end
        # next set of output plane base is doubled
        baseplanes *= 2
    end

    return Chain(Chain(layers...),
                Chain(AdaptiveMeanPool((1, 1)), flatten, Dense(inplanes, nclasses)))
end

struct ResNeXt
    layers
end

function ResNeXt(channel_config, block_config, cardinality; nclasses = 1000)
    layers = resnext(;channel_config = channel_config,
                  block_config = block_config,
                  cardinality = cardinality,
                  nclasses = nclasses)
    ResNeXt(layers)
end

@functor ResNeXt

(m::ResNeXt)(x) = m.layers(x)

backbone(m::ResNeXt) = m.layers[1]
classifier(m::ResNeXt) = m.layers[2]

function ResNeXt50(; pretrain = false, nclasses = 1000)
    channel_config = [1, 2, 1]
    block_config = [3, 4, 6, 3]
    cardinality = 32
    model = ResNeXt(channel_config, block_config, cardinality; nclasses)
    return model
end

Base.show(io::IO, ::MIME"text/plain", model::ResNeXt) = Metalhead._maybe_big_show(io, model)

m = ResNeXt50()
Flux._big_show(IO, m)