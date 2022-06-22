function drop_blocks(drop_prob = 0.0)
    return [
        identity,
        identity,
        DropBlock(drop_prob, 5, 0.25),
        DropBlock(drop_prob, 3, 1.00),
    ]
end

function downsample_conv(kernel_size, in_channels, out_channels; stride = 1, dilation = 1,
                         first_dilation = nothing, norm_layer = BatchNorm)
    kernel_size = stride == 1 && dilation == 1 ? 1 : kernel_size
    first_dilation = kernel_size[1] > 1 ?
                     (!isnothing(first_dilation) ? first_dilation : dilation) : 1
    pad = ((stride - 1) + dilation * (kernel_size[1] - 1)) ÷ 2
    return Chain(Conv(kernel_size, in_channels => out_channels; stride, pad,
                      dilation = first_dilation, bias = false),
                 norm_layer(out_channels))
end

function downsample_avg(kernel_size, in_channels, out_channels; stride = 1, dilation = 1,
                        first_dilation = nothing, norm_layer = BatchNorm)
    avg_stride = dilation == 1 ? stride : 1
    if stride == 1 && dilation == 1
        pool = identity
    else
        pad = avg_stride == 1 && dilation > 1 ? SamePad() : 0
        pool = avg_pool_fn((2, 2); stride = avg_stride, pad)
    end
    return Chain(pool,
                 Conv((1, 1), in_channels => out_channels; bias = false),
                 norm_layer(out_channels))
end

function basicblock(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
                    base_width = 64, reduce_first = 1, dilation = 1,
                    first_dilation = nothing, activation = relu, norm_layer = BatchNorm,
                    drop_block = identity, drop_path = identity)
    expansion = expansion_factor(basicblock)
    @assert cardinality==1 "BasicBlock only supports cardinality of 1"
    @assert base_width==64 "BasicBlock does not support changing base width"
    first_planes = planes ÷ reduce_first
    outplanes = planes * expansion
    first_dilation = !isnothing(first_dilation) ? first_dilation : dilation
    conv_bn1 = Chain(Conv((3, 3), inplanes => first_planes; stride, pad = first_dilation,
                          dilation = first_dilation, bias = false),
                     norm_layer(first_planes))
    drop_block = drop_block === identity ? identity : drop_block
    conv_bn2 = Chain(Conv((3, 3), first_planes => outplanes; stride, pad = dilation,
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
                     norm_layer(first_planes))
    conv_bn2 = Chain(Conv((3, 3), first_planes => width; stride, pad = first_dilation,
                          dilation = first_dilation, groups = cardinality, bias = false),
                     norm_layer(width))
    drop_block = drop_block === identity ? identity : drop_block()
    conv_bn3 = Chain(Conv((1, 1), width => outplanes; bias = false), norm_layer(outplanes))
    return Chain(Parallel(+, downsample,
                          Chain(conv_bn1, drop_block, activation, conv_bn2, drop_block,
                                activation, conv_bn3, drop_path)),
                 activation)
end
expansion_factor(::typeof(bottleneck)) = 4

function make_blocks(block_fn, channels, block_repeats, inplanes;
                     reduce_first = 1, output_stride = 32, down_kernel_size = 1,
                     avg_down = false, drop_block_rate = 0.0, drop_path_rate = 0.0,
                     kwargs...)
    expansion = expansion_factor(block_fn)
    kwarg_dict = Dict(kwargs...)
    stages = []
    net_block_idx = 1
    net_stride = 4
    dilation = prev_dilation = 1
    for (stage_idx, (planes, num_blocks, drop_block)) in enumerate(zip(channels,
                                                                       block_repeats,
                                                                       drop_blocks(drop_block_rate)))
        stride = stage_idx == 1 ? 1 : 2
        if net_stride >= output_stride
            dilation *= stride
            stride = 1
        else
            net_stride *= stride
        end
        # first block needs to be handled differently for downsampling
        downsample = identity
        if stride != 1 || inplanes != planes * expansion
            downsample = avg_down ?
                         downsample_avg(down_kernel_size, inplanes, planes * expansion;
                                        stride, dilation, first_dilation = prev_dilation,
                                        norm_layer = kwarg_dict[:norm_layer]) :
                         downsample_conv(down_kernel_size, inplanes, planes * expansion;
                                         stride, dilation, first_dilation = prev_dilation,
                                         norm_layer = kwarg_dict[:norm_layer])
        end
        block_kwargs = Dict(:reduce_first => reduce_first, :dilation => dilation,
                            :drop_block => drop_block, kwargs...)
        blocks = []
        for block_idx in 1:num_blocks
            downsample = block_idx == 1 ? downsample : identity
            stride = block_idx == 1 ? stride : 1
            # stochastic depth linear decay rule
            block_dpr = drop_path_rate * net_block_idx / (sum(block_repeats) - 1)
            push!(blocks,
                  block_fn(inplanes, planes; stride, downsample,
                           first_dilation = prev_dilation,
                           drop_path = DropPath(block_dpr), block_kwargs...))
            prev_dilation = dilation
            inplanes = planes * expansion
            net_block_idx += 1
        end
        push!(stages, Chain(blocks...))
    end
    return Chain(stages...)
end

function resnet(block, layers; num_classes = 1000, inchannels = 3, output_stride = 32,
                cardinality = 1, base_width = 64, stem_width = 64, stem_type = :default,
                replace_stem_pool = false, reduce_first = 1, down_kernel_size = (1, 1),
                avg_down = false, activation = relu, norm_layer = BatchNorm,
                drop_rate = 0.0, drop_path_rate = 0.0, drop_block_rate = 0.0,
                block_kwargs...)
    @assert output_stride in (8, 16, 32) "Invalid `output_stride`. Must be one of (8, 16, 32)"
    @assert stem_type in [:default, :deep, :deep_tiered] "Stem type must be one of [:default, :deep, :deep_tiered]"
    # Stem
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
    stem = Chain(conv1, bn1, stempool)
    # Feature Blocks
    channels = [64, 128, 256, 512]
    stage_blocks = make_blocks(block, channels, layers, inplanes; cardinality, base_width,
                               output_stride, reduce_first, avg_down,
                               down_kernel_size, activation, norm_layer,
                               drop_block_rate, drop_path_rate, block_kwargs...)
    # Head (Pooling and Classifier)
    expansion = expansion_factor(block)
    num_features = 512 * expansion
    classifier = Chain(GlobalMeanPool(), Dropout(drop_rate), MLUtils.flatten,
                       Dense(num_features, num_classes))

    return Chain(Chain(stem, stage_blocks), classifier)
end
