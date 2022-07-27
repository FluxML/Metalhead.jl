"""
    basicblock(inplanes, planes; stride = 1, downsample = identity,
               reduction_factor = 1, dilation = 1, first_dilation = dilation,
               activation = relu, connection = addact\$activation,
               norm_layer = BatchNorm, drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity)

Creates a basic ResNet block.

# Arguments

  - `inplanes`: number of input feature maps
  - `planes`: number of feature maps for the block
  - `stride`: the stride of the block
  - `downsample`: the downsampling function to use
  - `reduction_factor`: the reduction factor that the input feature maps are reduced by before the first
    convolution.
  - `connection`: the function applied to the output of residual and skip paths in
    a block. See [`addact`](#) and [`actadd`](#) for an example.
  - `activation`: the activation function to use.
  - `norm_layer`: the normalization layer to use.
  - `drop_block`: the drop block layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `drop_path`: the drop path layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
"""
function basicblock(inplanes::Integer, planes::Integer; downsample_fns,
                    stride::Integer = 1, reduction_factor::Integer = 1,
                    connection = addact, activation = relu,
                    norm_layer = BatchNorm, prenorm::Bool = false,
                    drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity)
    expansion = expansion_factor(basicblock)
    first_planes = planes ÷ reduction_factor
    outplanes = planes * expansion
    conv_bn1 = conv_norm((3, 3), inplanes => first_planes, identity; norm_layer, prenorm,
                         stride, pad = 1, bias = false)
    conv_bn2 = conv_norm((3, 3), first_planes => outplanes, identity; norm_layer, prenorm,
                         pad = 1, bias = false)
    layers = [conv_bn1..., drop_block, activation, conv_bn2..., attn_fn(outplanes),
        drop_path]
    downsample = downsample_block(downsample_fns, inplanes, planes, expansion;
                                  stride, norm_layer, prenorm)
    return Parallel(connection$activation, Chain(filter!(!=(identity), layers)...),
                    downsample)
end
expansion_factor(::typeof(basicblock)) = 1

"""
    bottleneck(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
               base_width = 64, reduction_factor = 1, first_dilation = 1,
               activation = relu, connection = addact\$activation,
               norm_layer = BatchNorm, drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity)

Creates a bottleneck ResNet block.

# Arguments

  - `inplanes`: number of input feature maps
  - `planes`: number of feature maps for the block
  - `stride`: the stride of the block
  - `downsample`: the downsampling function to use
  - `cardinality`: the number of groups in the convolution.
  - `base_width`: the number of output feature maps for each convolutional group.
  - `reduction_factor`: the reduction factor that the input feature maps are reduced by before the first
    convolution.
  - `activation`: the activation function to use.
  - `connection`: the function applied to the output of residual and skip paths in
    a block. See [`addact`](#) and [`actadd`](#) for an example.
  - `norm_layer`: the normalization layer to use.
  - `drop_block`: the drop block layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `drop_path`: the drop path layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
"""
function bottleneck(inplanes::Integer, planes::Integer; downsample_fns, stride::Integer = 1,
                    cardinality::Integer = 1, base_width::Integer = 64,
                    reduction_factor::Integer = 1, connection = addact, activation = relu,
                    norm_layer = BatchNorm, prenorm::Bool = false,
                    drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity)
    expansion = expansion_factor(bottleneck)
    width = floor(Int, planes * (base_width / 64)) * cardinality
    first_planes = width ÷ reduction_factor
    outplanes = planes * expansion
    conv_bn1 = conv_norm((1, 1), inplanes => first_planes, activation; norm_layer, prenorm,
                         bias = false)
    conv_bn2 = conv_norm((3, 3), first_planes => width, identity; norm_layer, prenorm,
                         stride, pad = 1, groups = cardinality, bias = false)
    conv_bn3 = conv_norm((1, 1), width => outplanes, identity; norm_layer, prenorm,
                         bias = false)
    downsample = downsample_block(downsample_fns, inplanes, planes, expansion;
                                  stride, norm_layer, prenorm)
    layers = [conv_bn1..., conv_bn2..., drop_block, activation, conv_bn3...,
        attn_fn(outplanes), drop_path]
    return Parallel(connection$activation, Chain(filter!(!=(identity), layers)...),
                    downsample)
end
expansion_factor(::typeof(bottleneck)) = 4

# Downsample layer using convolutions.
function downsample_conv(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm, prenorm = false)
    return Chain(conv_norm((1, 1), inplanes => outplanes, identity; norm_layer, prenorm,
                           pad = SamePad(), stride, bias = false)...)
end

# Downsample layer using max pooling
function downsample_pool(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm, prenorm = false)
    pool = (stride == 1) ? identity : MeanPool((2, 2); stride, pad = SamePad())
    return Chain(pool,
                 conv_norm((1, 1), inplanes => outplanes, identity; norm_layer, prenorm,
                           bias = false)...)
end

# Downsample layer which is an identity projection. Uses max pooling
# when the output size is more than the input size.
function downsample_identity(inplanes::Integer, outplanes::Integer; kwargs...)
    if outplanes > inplanes
        return Chain(MaxPool((1, 1); stride = 2),
                     y -> cat_channels(y,
                                       zeros(eltype(y),
                                             size(y, 1),
                                             size(y, 2),
                                             outplanes - inplanes, size(y, 4))))
    else
        return identity
    end
end

function downsample_block(downsample_fns, inplanes, planes, expansion; stride, kwargs...)
    down_fn = (stride != 1 || inplanes != planes * expansion) ? downsample_fns[1] :
              downsample_fns[2]
    return down_fn(inplanes, planes * expansion; stride, kwargs...)
end

# Shortcut configurations for the ResNet models
const shortcut_dict = Dict(:A => (downsample_identity, downsample_identity),
                           :B => (downsample_conv, downsample_identity),
                           :C => (downsample_conv, downsample_conv),
                           :D => (downsample_pool, downsample_identity))

# Makes the downsample `Vector`` with `NTuple{2}`s of functions when it is
# specified as a `Vector` of `Symbol`s. This is used to make the downsample
# `Vector` for the `_make_blocks` function. If the `eltype(::Vector)` is
# already an `NTuple{2}` of functions, it is returned unchanged.
function _make_downsample_fns(vec::Vector{<:Symbol}, block_repeats)
    downs = []
    for i in vec
        @assert i in keys(shortcut_dict)
        "The shortcut type must be one of $(sort(collect(keys(shortcut_dict))))"
        push!(downs, shortcut_dict[i])
    end
    return downs
end
function _make_downsample_fns(sym::Symbol, block_repeats)
    @assert sym in keys(shortcut_dict)
    "The shortcut type must be one of $(sort(collect(keys(shortcut_dict))))"
    return collect(shortcut_dict[sym] for _ in 1:length(block_repeats))
end
_make_downsample_fns(vec::Vector{<:NTuple{2}}, block_repeats) = vec
_make_downsample_fns(tup::NTuple{2}, block_repeats) = [tup for _ in 1:length(block_repeats)]

# Stride for each block in the ResNet model
function get_stride(stage_idx::Integer, block_idx::Integer)
    return (stage_idx == 1 || block_idx != 1) ? 1 : 2
end

# returns `DropBlock`s for each stage of the ResNet as in timm.
# TODO - add experimental options for DropBlock as part of the API (#188)
function _drop_blocks(drop_block_rate::AbstractFloat)
    return [
        identity, identity,
        DropBlock(drop_block_rate, 5, 0.25), DropBlock(drop_block_rate, 3, 1.00),
    ]
end

"""
    resnet_stem(; stem_type = :default, inchannels = 3, replace_stem_pool = false,
                  norm_layer = BatchNorm, activation = relu)

Builds a stem to be used in a ResNet model. See the `stem` argument of [`resnet`](#) for details
on how to use this function.

# Arguments

  - `stem_type`: The type of stem to be built. One of `[:default, :deep, :deep_tiered]`.
    
      + `:default`: Builds a stem based on the default ResNet stem, which consists of a single
        7x7 convolution with stride 2 and a normalisation layer followed by a 3x3 max pooling
        layer with stride 2.
      + `:deep`: This borrows ideas from other papers (InceptionResNet-v2, for example) in using
        a deeper stem with 3 successive 3x3 convolutions having normalisation layers after each
        one. This is followed by a 3x3 max pooling layer with stride 2.
      + `:deep_tiered`: A variant of the `:deep` stem that has a larger width in the second
        convolution. This is an experimental variant from the `timm` library in Python that
        shows peformance improvements over the `:deep` stem in some cases.

  - `inchannels`: The number of channels in the input.
  - `replace_pool`: Set to true to replace the max pooling layers with a 3x3 convolution + normalization with a stride of two.
  - `norm_layer`: The normalisation layer used in the stem.
  - `activation`: The activation function used in the stem.
"""
function resnet_stem(stem_type::Symbol = :default; inchannels::Integer = 3,
                     replace_pool::Bool = false, norm_layer = BatchNorm, prenorm = false,
                     activation = relu)
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
        conv1 = Chain(conv_norm((3, 3), inchannels => stem_channels[1], activation;
                                norm_layer, prenorm, stride = 2, pad = 1, bias = false)...,
                      conv_norm((3, 3), stem_channels[1] => stem_channels[2], activation;
                                norm_layer, pad = 1, bias = false)...,
                      Conv((3, 3), stem_channels[2] => inplanes; pad = 1, bias = false))
    else
        conv1 = Conv((7, 7), inchannels => inplanes; stride = 2, pad = 3, bias = false)
    end
    bn1 = norm_layer(inplanes, activation)
    # Stem pooling
    stempool = replace_pool ?
               Chain(conv_norm((3, 3), inplanes => inplanes, activation; norm_layer,
                               prenorm, stride = 2, pad = 1, bias = false)...) :
               MaxPool((3, 3); stride = 2, pad = 1)
    return Chain(conv1, bn1, stempool), inplanes
end

function block_args(::typeof(basicblock), block_repeats;
                    downsample_vec, reduction_factor = 1, activation = relu,
                    norm_layer = BatchNorm, prenorm = false,
                    drop_path_rate = 0.0, drop_block_rate = 0.0,
                    attn_fn = planes -> identity)
    pathschedule = linear_scheduler(drop_path_rate; depth = sum(block_repeats))
    blockschedule = linear_scheduler(drop_block_rate; depth = sum(block_repeats))
    function get_layers(stage_idx, block_idx)
        stride = get_stride(stage_idx, block_idx)
        downsample_fns = downsample_vec[stage_idx]
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        drop_path = DropPath(pathschedule[schedule_idx])
        drop_block = DropBlock(blockschedule[schedule_idx])
        return (; downsample_fns, reduction_factor, stride, activation, norm_layer,
                prenorm, drop_path, drop_block, attn_fn)
    end
end

function block_args(::typeof(bottleneck), block_repeats;
                    downsample_vec, cardinality = 1, base_width = 64,
                    reduction_factor = 1, activation = relu,
                    norm_layer = BatchNorm, prenorm = false,
                    drop_block_rate = 0.0, drop_path_rate = 0.0,
                    attn_fn = planes -> identity)
    pathschedule = linear_scheduler(drop_path_rate; depth = sum(block_repeats))
    blockschedule = linear_scheduler(drop_block_rate; depth = sum(block_repeats))
    function get_layers(stage_idx, block_idx)
        stride = get_stride(stage_idx, block_idx)
        downsample_fns = downsample_vec[stage_idx]
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        drop_path = DropPath(pathschedule[schedule_idx])
        drop_block = DropBlock(blockschedule[schedule_idx])
        return (; downsample_fns, reduction_factor, cardinality, base_width, stride,
                activation, norm_layer, prenorm, drop_path, drop_block, attn_fn)
    end
end

# Makes the main stages of the ResNet model. This is an internal function and should not be 
# used by end-users. `block_fn` is a function that returns a single block of the ResNet. 
# See `basicblock` and `bottleneck` for examples. A block must define a function 
# `expansion(::typeof(block))` that returns the expansion factor of the block.
function resnet_stages(block_fn, block_repeats::Vector{<:Integer}, inplanes::Integer;
                       kwargs...)
    # Construct each stage
    stages = []
    for (stage_idx, (num_blocks)) in enumerate(block_repeats)
        planes = 64 * 2^(stage_idx - 1)
        get_kwargs = block_args(block_fn, block_repeats; kwargs...)
        # Construct the blocks for each stage
        blocks = []
        for block_idx in range(1, num_blocks)
            push!(blocks, block_fn(inplanes, planes; get_kwargs(stage_idx, block_idx)...))
            inplanes = planes * expansion_factor(block_fn)
        end
        push!(stages, Chain(blocks...))
    end
    return Chain(stages...)
end

function resnet(block_fn, block_repeats::Vector{<:Integer}, downsample_opt = :B;
                imsize::Dims{2} = (256, 256), inchannels::Integer = 3,
                stem = first(resnet_stem(; inchannels)), inplanes::Integer = 64,
                pool_layer = AdaptiveMeanPool((1, 1)), dropout_rate = 0.0,
                use_conv_classifier::Bool = false, nclasses::Integer = 1000, kwargs...)
    # Configure downsample templates
    downsample_vec = _make_downsample_fns(downsample_opt, block_repeats)
    # Build stages of the ResNet
    stage_blocks = resnet_stages(block_fn, block_repeats, inplanes; downsample_vec,
                                 kwargs...)
    backbone = Chain(stem, stage_blocks)
    # Build the classifier head
    outfeatures = Flux.outputsize(backbone, (imsize..., inchannels); padbatch = true)
    classifier = create_classifier(outfeatures[3], nclasses; dropout_rate, pool_layer,
                                   use_conv = use_conv_classifier)
    return Chain(backbone, classifier)
end

# block-layer configurations for ResNet-like models
const resnet_configs = Dict(18 => (basicblock, [2, 2, 2, 2]),
                            34 => (basicblock, [3, 4, 6, 3]),
                            50 => (bottleneck, [3, 4, 6, 3]),
                            101 => (bottleneck, [3, 4, 23, 3]),
                            152 => (bottleneck, [3, 8, 36, 3]))
