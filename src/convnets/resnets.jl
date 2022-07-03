function downsample_conv(kernel_size, inplanes, outplanes; stride = 1, dilation = 1,
                         first_dilation = nothing, norm_layer = BatchNorm)
    kernel_size = stride == 1 && dilation == 1 ? (1, 1) : kernel_size
    first_dilation = kernel_size[1] > 1 ?
                     (!isnothing(first_dilation) ? first_dilation : dilation) : 1
    pad = ((stride - 1) + dilation * (kernel_size[1] - 1)) ÷ 2
    return Chain(Conv(kernel_size, inplanes => outplanes; stride, pad,
                      dilation = first_dilation, bias = false),
                 norm_layer(outplanes))
end

function downsample_avg(kernel_size, inplanes, outplanes; stride = 1, dilation = 1,
                        first_dilation = nothing, norm_layer = BatchNorm)
    avg_stride = dilation == 1 ? stride : 1
    if stride == 1 && dilation == 1
        pool = identity
    else
        pad = avg_stride == 1 && dilation > 1 ? SamePad() : 0
        pool = MeanPool((2, 2); stride = avg_stride, pad)
    end
    return Chain(pool,
                 Conv((1, 1), inplanes => outplanes; bias = false),
                 norm_layer(outplanes))
end

function basicblock(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
                    base_width = 64, reduce_first = 1, dilation = 1,
                    first_dilation = nothing, activation = relu, norm_layer = BatchNorm,
                    drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity, attn_args::NamedTuple = NamedTuple())
    expansion = expansion_factor(basicblock)
    @assert cardinality==1 "`basicblock` only supports cardinality of 1"
    @assert base_width==64 "`basicblock` does not support changing base width"
    first_planes = planes ÷ reduce_first
    outplanes = planes * expansion
    first_dilation = !isnothing(first_dilation) ? first_dilation : dilation
    conv_bn1 = Chain(Conv((3, 3), inplanes => first_planes; stride, pad = first_dilation,
                          dilation = first_dilation, bias = false),
                     norm_layer(first_planes))
    drop_block = drop_block
    conv_bn2 = Chain(Conv((3, 3), first_planes => outplanes; pad = dilation,
                          dilation = dilation, bias = false),
                     norm_layer(outplanes))
    attn_layer = attn_fn(outplanes; attn_args...)
    return Chain(Parallel(+, downsample,
                          Chain(conv_bn1, drop_block, activation, conv_bn2, attn_layer,
                                drop_path)),
                 activation)
end
expansion_factor(::typeof(basicblock)) = 1

function bottleneck(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
                    base_width = 64, reduce_first = 1, dilation = 1,
                    first_dilation = nothing, activation = relu, norm_layer = BatchNorm,
                    drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity, attn_args::NamedTuple = NamedTuple())
    expansion = expansion_factor(bottleneck)
    width = floor(Int, planes * (base_width / 64)) * cardinality
    first_planes = width ÷ reduce_first
    outplanes = planes * expansion
    first_dilation = !isnothing(first_dilation) ? first_dilation : dilation
    conv_bn1 = Chain(Conv((1, 1), inplanes => first_planes; bias = false),
                     norm_layer(first_planes, activation))
    conv_bn2 = Chain(Conv((3, 3), first_planes => width; stride, pad = first_dilation,
                          dilation = first_dilation, groups = cardinality, bias = false),
                     norm_layer(width))
    conv_bn3 = Chain(Conv((1, 1), width => outplanes; bias = false), norm_layer(outplanes))
    attn_layer = attn_fn(outplanes; attn_args...)
    return Chain(Parallel(+, downsample,
                          Chain(conv_bn1, conv_bn2, drop_block, activation, conv_bn3,
                                attn_layer, drop_path)),
                 activation)
end
expansion_factor(::typeof(bottleneck)) = 4

"""
    resnet_stem(; stem_type = :default, inchannels = 3, replace_stem_pool = false,
                  norm_layer = BatchNorm, activation = relu)

Builds a stem to be used in a ResNet model. See the `stem` argument of `resnet` for details
on how to use this function.

# Arguments:

  - `stem_type`: The type of stem to be built. One of `[:default, :deep, :deep_tiered]`.
    
      + `:default`: Builds a stem based on the default ResNet stem, which consists of a single
        7x7 convolution with stride 2 and a normalisation layer followed by a 3x3
        max pooling layer with stride 2.
      + `:deep`: This borrows ideas from other papers (InceptionResNet-v2 for one) in using a
        deeper stem with 3 successive 3x3 convolutions having normalisation layers
        after each one. This is followed by a 3x3 max pooling layer with stride 2.
      + `:deep_tiered`: A variant of the `:deep` stem that has a larger width in the second
        convolution. This is an experimental variant from the `timm` library
        in Python that shows peformance improvements over the `:deep` stem
        in some cases.

  - `inchannels`: The number of channels in the input.
  - `replace_stem_pool`: Whether to replace the default 3x3 max pooling layer with a
    3x3 convolution with stride 2 and a normalisation layer.
  - `norm_layer`: The normalisation layer used in the stem.
  - `activation`: The activation function used in the stem.
"""
function resnet_stem(; stem_type = :default, inchannels = 3, replace_stem_pool = false,
                     norm_layer = BatchNorm, activation = relu)
    @assert stem_type in [:default, :deep, :deep_tiered]
    "Stem type must be one of [:default, :deep, :deep_tiered]"
    # Main stem
    deep_stem = stem_type == :deep || stem_type == :deep_tiered
    inplanes = deep_stem ? stem_width * 2 : 64
    # Deep stem that uses three successive 3x3 convolutions instead of a single 7x7 convolution
    if deep_stem
        if stem_type == :deep
            stem_channels = (stem_width, stem_width)
        elseif stem_type == :deep_tiered
            stem_channels = (3 * (stem_width ÷ 4), stem_width)
        end
        conv1 = Chain(Conv((3, 3), inchannels => stem_channels[1]; stride = 2, pad = 1,
                           bias = false),
                      norm_layer(stem_channels[1], activation),
                      Conv((3, 3), stem_channels[1] => stem_channels[1]; pad = 1,
                           bias = false),
                      norm_layer(stem_channels[2], activation),
                      Conv((3, 3), stem_channels[2] => inplanes; pad = 1, bias = false))
    else
        conv1 = Conv((7, 7), inchannels => inplanes; stride = 2, pad = 3, bias = false)
    end
    bn1 = norm_layer(inplanes, activation)
    # Stem pooling
    if replace_stem_pool
        stempool = Chain(Conv((3, 3), inplanes => inplanes; stride = 2, pad = 1,
                              bias = false),
                         norm_layer(inplanes, activation))
    else
        stempool = MaxPool((3, 3); stride = 2, pad = 1)
    end
    return Chain(conv1, bn1, stempool), inplanes
end

function downsample_block(downsample_fn, inplanes, planes, expansion; kernel_size = (1, 1),
                          stride = 1, dilation = 1, first_dilation = dilation,
                          norm_layer = BatchNorm)
    if stride != 1 || inplanes != planes * expansion
        downsample = downsample_fn(kernel_size, inplanes, planes * expansion;
                                   stride, dilation, first_dilation,
                                   norm_layer)
    else
        downsample = identity
    end
    return downsample
end

# Makes the main stages of the ResNet model. This is an internal function and should not be 
# used by end-users. `block_fn` is a function that returns a single block of the ResNet. 
# See `basicblock` and `bottleneck` for examples. A block must define a function 
# `expansion(::typeof(block))` that returns the expansion factor of the block.
function _make_blocks(block_fn, channels, block_repeats, inplanes; output_stride = 32,
                      downsample_fn = downsample_conv,
                      drop_rates::NamedTuple, block_args::NamedTuple)
    @assert output_stride in (8, 16, 32) "Invalid `output_stride`. Must be one of (8, 16, 32)"
    expansion = expansion_factor(block_fn)
    stages = []
    net_block_idx = 1
    net_stride = 4
    dilation = prev_dilation = 1
    # Stochastic depth linear decay rule (DropPath)
    dp_rates = LinRange{Float32}(0.0, get(drop_rates, :drop_path_rate, 0),
                                 sum(block_repeats))
    for (stage_idx, (planes, num_blocks, drop_block)) in enumerate(zip(channels,
                                                                       block_repeats,
                                                                       _drop_blocks(get(drop_rates,
                                                                                        :drop_block_rate,
                                                                                        0))))
        # Stride calculations for each stage
        stride = stage_idx == 1 ? 1 : 2
        if net_stride >= output_stride
            dilation *= stride
            stride = 1
        else
            net_stride *= stride
        end
        # Downsample block; either a (default) convolution-based block or a pooling-based block
        downsample = downsample_block(downsample_fn, inplanes, planes, expansion;
                                      stride, dilation, first_dilation = dilation)
        # Construct the blocks for each stage
        blocks = []
        for block_idx in 1:num_blocks
            downsample = block_idx == 1 ? downsample : identity
            stride = block_idx == 1 ? stride : 1
            push!(blocks,
                  block_fn(inplanes, planes; stride, downsample,
                           first_dilation = prev_dilation,
                           drop_path = DropPath(dp_rates[block_idx]), drop_block,
                           block_args...))
            prev_dilation = dilation
            inplanes = planes * expansion
            net_block_idx += 1
        end
        push!(stages, Chain(blocks...))
    end
    return Chain(stages...)
end

# returns `DropBlock`s for each stage of the ResNet
function _drop_blocks(drop_block_prob = 0.0)
    return [
        identity, identity,
        DropBlock(drop_block_prob, 5, 0.25), DropBlock(drop_block_prob, 3, 1.00),
    ]
end

function resnet(block_fn, layers; nclasses = 1000, inchannels = 3, output_stride = 32,
                stem = first(resnet_stem(; inchannels)), inplanes = 64,
                downsample_fn = downsample_conv, block_args::NamedTuple = NamedTuple(),
                drop_rates::NamedTuple = (dropout_rate = 0.0, drop_path_rate = 0.0,
                                          drop_block_rate = 0.0),
                classifier_args::NamedTuple = NamedTuple())
    # Feature Blocks
    channels = [64, 128, 256, 512]
    stage_blocks = _make_blocks(block_fn, channels, layers, inplanes;
                                output_stride, downsample_fn, drop_rates, block_args)
    # Head (Pooling and Classifier)
    expansion = expansion_factor(block_fn)
    num_features = 512 * expansion
    global_pool, fc = create_classifier(num_features, nclasses; classifier_args...)
    classifier = Chain(global_pool, Dropout(get(drop_rates, :dropout_rate, 0)), fc)
    return Chain(Chain(stem, stage_blocks), classifier)
end

# block-layer configurations for ResNet and ResNeXt models
const resnet_config = Dict(18 => (basicblock, [2, 2, 2, 2]),
                           34 => (basicblock, [3, 4, 6, 3]),
                           50 => (bottleneck, [3, 4, 6, 3]),
                           101 => (bottleneck, [3, 4, 23, 3]),
                           152 => (bottleneck, [3, 8, 36, 3]))
struct ResNet
    layers::Any
end
@functor ResNet

(m::ResNet)(x) = m.layers(x)

"""
    ResNet(depth::Integer; pretrain = false, nclasses = 1000) 

Creates a ResNet model with the specified depth.
((reference)[https://arxiv.org/abs/1512.03385])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `nclasses`: the number of output classes

!!! warning
    
    `ResNet` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](#).
"""
function ResNet(depth::Integer; pretrain = false, nclasses = 1000)
    @assert depth in [18, 34, 50, 101, 152]
    "Invalid depth. Must be one of [18, 34, 50, 101, 152]"
    layers = resnet(resnet_config[depth]...; nclasses)
    if pretrain
        loadpretrain!(layers, string("resnet", depth))
    end
    return ResNet(layers)
end

struct ResNeXt
    layers::Any
end
@functor ResNeXt

(m::ResNeXt)(x) = m.layers(x)

"""
    ResNeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4, nclasses = 1000) 

Creates a ResNeXt model with the specified depth, cardinality, and base width.
((reference)[https://arxiv.org/abs/1611.05431])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `cardinality`: the number of groups to be used in the 3x3 convolution in each block.
  - `base_width`: the number of feature maps in each group.
  - `nclasses`: the number of output classes

!!! warning
    
    `ResNeXt` does not currently support pretrained weights.

Advanced users who want more configuration options will be better served by using [`resnet`](#).
"""
function ResNeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4,
                 nclasses = 1000)
    @assert depth in [50, 101, 152]
    "Invalid depth. Must be one of [50, 101, 152]"
    layers = resnet(resnet_config[depth]...; nclasses,
                    block_args = (; cardinality, base_width))
    if pretrain
        loadpretrain!(layers, string("resnext", depth, "_", cardinality, "x", base_width))
    end
    return ResNeXt(layers)
end

struct SEResNet
    layers::Any
end
@functor SEResNet

(m::SEResNet)(x) = m.layers(x)

"""
    SEResNet(depth::Integer; pretrain = false, nclasses = 1000)

Creates a SEResNet model with the specified depth.
((reference)[https://arxiv.org/pdf/1709.01507.pdf])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `nclasses`: the number of output classes
"""
function SEResNet(depth::Integer; pretrain = false, nclasses = 1000)
    @assert depth in [18, 34, 50, 101, 152]
    "Invalid depth. Must be one of [18, 34, 50, 101, 152]"
    layers = resnet(resnet_config[depth]...; nclasses,
                    block_args = (; attn_fn = squeeze_excite))
    if pretrain
        loadpretrain!(layers, string("seresnet", depth))
    end
    return SEResNet(layers)
end

struct SEResNeXt
    layers::Any
end
@functor SEResNeXt

(m::SEResNeXt)(x) = m.layers(x)

"""
    SEResNeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4, nclasses = 1000)

Creates a SEResNeXt model with the specified depth, cardinality, and base width.
((reference)[https://arxiv.org/pdf/1709.01507.pdf])

# Arguments

  - `depth`: one of `[18, 34, 50, 101, 152]`. The depth of the ResNet model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `cardinality`: the number of groups to be used in the 3x3 convolution in each block.
  - `base_width`: the number of feature maps in each group.
  - `nclasses`: the number of output classes
"""
function SEResNeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4,
                   nclasses = 1000)
    @assert depth in [50, 101, 152]
    "Invalid depth. Must be one of [50, 101, 152]"
    layers = resnet(resnet_config[depth]...; nclasses,
                    block_args = (; cardinality, base_width, attn_fn = squeeze_excite))
    if pretrain
        loadpretrain!(layers, string("seresnext", depth, "_", cardinality, "x", base_width))
    end
    return SEResNeXt(layers)
end
