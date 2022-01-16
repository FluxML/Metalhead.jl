"""
    resnextblock(inplanes, outplanes, cardinality, downsample = false)

Create a basic residual block as defined in the paper for ResNeXt
([reference](https://arxiv.org/abs/1611.05431)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: a list of the number of output feature maps for each convolution
               within the residual block
- `cardinality`: the number of groups to use for the convolution
- `downsample`: set to `true` to downsample the input
"""
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

"""
    resnext(connection = (x, y) -> @. relu(x) + relu(y); channel_config, block_config, 
            cardinality, nclasses = 1000)
    
Create a ResNeXt model
([reference](https://arxiv.org/abs/1611.05431)).

# Arguments
- `connection`: the binary function applied to the output of residual and skip paths in a block
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `cardinality`: the number of groups to use for the convolution
- `nclasses`: the number of output classes
"""
function resnext(cardinality, width, channel_multiplier = 2, connection = (x, y) -> @. relu(x) + relu(y);
                 block_config, nclasses = 1000)
    inplanes = 64
    baseplanes = 128
    layers = []
    append!(layers, conv_bn((7, 7), 3, inplanes; stride = 2, pad = (3, 3)))
    push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))
    for (i, nrepeats) in enumerate(block_config)
        # output planes within a block
        outplanes = baseplanes * channel_multiplier
        # push first skip connection on using first residual
        # downsample the residual path if this is the first repetition of a block
        push!(layers, Parallel(connection, resnextblock(inplanes, outplanes, cardinality, width, i != 1),
                                           skip_projection(inplanes, outplanes, i != 1)))
        # push remaining skip connections on using second residual
        inplanes = outplanes
        for _ in 2:nrepeats
            push!(layers, Parallel(connection, resnextblock(inplanes, outplanes, cardinality, width, false),
                                               skip_identity(inplanes, outplanes, false)))
        end
        baseplanes = outplanes
    end

    return Chain(Chain(layers...),
                 Chain(AdaptiveMeanPool((1, 1)), flatten, Dense(inplanes, nclasses)))
end

"""
    ResNeXt(channel_config, block_config, cardinality; nclasses = 1000)
    
Create a ResNeXt model
([reference](https://arxiv.org/abs/1611.05431)).

# Arguments
- `channel_config`: the growth rate of the output feature maps within a residual block
- `block_config`: a list of the number of residual blocks at each stage
- `cardinality`: the number of groups to use for the convolution
- `nclasses`: the number of output classes
"""
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

"""
    ResNeXt50(; pretrain = false, nclasses = 1000)
   
Create a ResNeXt-50 model
([reference](https://arxiv.org/abs/1611.05431)).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNeXt50` does not currently support pretrained weights.
"""
function ResNeXt50(; pretrain = false, nclasses = 1000)
    channel_config = [1, 1, 2]
    block_config = [3, 4, 6, 3]
    cardinality = 32
    return ResNeXt(channel_config, block_config, cardinality; nclasses)
end

"""
    ResNeXt101(; pretrain = false, nclasses = 1000)

Create a ResNeXt-101 model
([reference](https://arxiv.org/abs/1611.05431)).

# Arguments
- `pretrain`: set to `true` to load pre-trained weights for ImageNet
- `nclasses`: the number of output classes

!!! warning
    `ResNeXt101` does not currently support pretrained weights.
"""
function ResNeXt101(; pretrain = false, nclasses = 1000)
    channel_config = [1, 1, 2]
    block_config = [3, 4, 23, 3]
    cardinality = 32
    return ResNeXt(channel_config, block_config, cardinality; nclasses)
end