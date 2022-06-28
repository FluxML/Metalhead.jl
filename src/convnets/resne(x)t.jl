# returns `DropBlock`s for each block of the ResNet
function _drop_blocks(drop_block_prob = 0.0)
    return [
        identity,
        identity,
        DropBlock(drop_block_prob, 5, 0.25),
        DropBlock(drop_block_prob, 3, 1.00),
    ]
end

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
                    drop_block = identity, drop_path = identity)
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
    return Chain(Parallel(+, downsample,
                          Chain(conv_bn1, drop_block, activation, conv_bn2, drop_path)),
                 activation)
end
expansion_factor(::typeof(basicblock)) = 1

function bottleneck(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
                    base_width = 64, reduce_first = 1, dilation = 1,
                    first_dilation = nothing, activation = relu, norm_layer = BatchNorm,
                    drop_block = identity, drop_path = identity)
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
    return Chain(Parallel(+, downsample,
                          Chain(conv_bn1, conv_bn2, drop_block, activation, conv_bn3,
                                drop_path)),
                 activation)
end
expansion_factor(::typeof(bottleneck)) = 4

function resnet_stem(; stem_type = :default, inchannels = 3, replace_stem_pool = false,
                     norm_layer = BatchNorm, activation = relu)
    @assert stem_type in [:default, :deep, :deep_tiered] "Stem type must be one of [:default, :deep, :deep_tiered]"
    # Main stem
    inplanes = stem_type == :deep ? stem_width * 2 : 64
    if stem_type == :deep
        stem_channels = (stem_width, stem_width)
        if stem_type == :deep_tiered
            stem_channels = (3 * (stem_width ÷ 4), stem_width)
        end
        conv1 = Chain(Conv((3, 3), inchannels => stem_channels[0]; stride = 2, pad = 1,
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
    return Chain(conv1, bn1, stempool)
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
                      downsample_fn = downsample_conv, downsample_args::NamedTuple = (),
                      drop_block_rate = 0.0, drop_path_rate = 0.0,
                      block_args::NamedTuple = ())
    @assert output_stride in (8, 16, 32) "Invalid `output_stride`. Must be one of (8, 16, 32)"
    expansion = expansion_factor(block_fn)
    stages = []
    net_block_idx = 1
    net_stride = 4
    dilation = prev_dilation = 1
    for (stage_idx, (planes, num_blocks, drop_block)) in enumerate(zip(channels,
                                                                       block_repeats,
                                                                       _drop_blocks(drop_block_rate)))
        # Stride calculations for each stage
        stride = stage_idx == 1 ? 1 : 2
        if net_stride >= output_stride
            dilation *= stride
            stride = 1
        else
            net_stride *= stride
        end
        # Downsample block; either a (default) convolution-based block or a pooling-based block.
        downsample = downsample_block(downsample_fn, inplanes, planes, expansion;
                                      downsample_args...)
        # Construct the blocks for each stage
        blocks = []
        for block_idx in 1:num_blocks
            downsample = block_idx == 1 ? downsample : identity
            stride = block_idx == 1 ? stride : 1
            # stochastic depth linear decay rule
            block_dpr = drop_path_rate * net_block_idx / (sum(block_repeats) - 1)
            push!(blocks,
                  block_fn(inplanes, planes; stride, downsample,
                           first_dilation = prev_dilation,
                           drop_path = DropPath(block_dpr), drop_block, block_args...))
            prev_dilation = dilation
            inplanes = planes * expansion
            net_block_idx += 1
        end
        push!(stages, Chain(blocks...))
    end
    return Chain(stages...)
end

function resnet(block, layers; nclasses = 1000, inchannels = 3, output_stride = 32,
                stem_fn = resnet_stem, stem_args::NamedTuple = (),
                downsample_fn = downsample_conv, downsample_args::NamedTuple = (),
                drop_rates::NamedTuple = (drop_rate = 0.0, drop_path_rate = 0.0,
                                          drop_block_rate = 0.0),
                block_args::NamedTuple = ())
    # Stem
    stem = stem_fn(; inchannels, stem_args...)
    # Feature Blocks
    channels = [64, 128, 256, 512]
    stage_blocks = _make_blocks(block, channels, layers, inchannels;
                                output_stride, downsample_fn, downsample_args,
                                drop_block_rate = drop_rates.drop_block_rate,
                                drop_path_rate = drop_rates.drop_path_rate,
                                block_args)
    # Head (Pooling and Classifier)
    expansion = expansion_factor(block)
    num_features = 512 * expansion
    classifier = Chain(GlobalMeanPool(), Dropout(drop_rates.drop_rate), MLUtils.flatten,
                       Dense(num_features, nclasses))
    return Chain(Chain(stem, stage_blocks), classifier)
end

const resnet_config = Dict(18 => (basicblock, [2, 2, 2, 2]),
                           34 => (basicblock, [3, 4, 6, 3]),
                           50 => (bottleneck, [3, 4, 6, 3]),
                           101 => (bottleneck, [3, 4, 23, 3]),
                           152 => (bottleneck, [3, 8, 36, 3]))
struct ResNet
    layers::Any
end
@functor ResNet

function ResNet(depth::Integer; pretrain = false, nclasses = 1000, kwargs...)
    @assert depth in [18, 34, 50, 101, 152] "Invalid depth. Must be one of [18, 34, 50, 101, 152]"
    model = resnet(resnet_config[depth]...; nclasses, kwargs...)
    if pretrain
        loadpretrain!(model, string("resnet", depth))
    end
    return model
end
