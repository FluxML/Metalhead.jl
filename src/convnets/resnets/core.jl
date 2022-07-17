## It is recommended to check out the user guide for more information.

abstract type AbstractResNetBlock end

struct basicblock <: AbstractResNetBlock
    inplanes::Integer
    planes::Integer
    reduction_factor::Integer
end
function basicblock(inplanes, planes, reduction_factor, base_width, cardinality)
    @assert base_width == 64 "`base_width` must be 64 for `basicblock`"
    @assert cardinality == 1 "`cardinality` must be 1 for `basicblock`"
    return basicblock(inplanes, planes, reduction_factor)
end
expansion_factor(::basicblock) = 1

struct bottleneck <: AbstractResNetBlock
    inplanes::Integer
    planes::Integer
    reduction_factor::Integer
    base_width::Integer
    cardinality::Integer
end
expansion_factor(::bottleneck) = 4

# Downsample layer using convolutions.
function downsample_conv(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm)
    return Chain(Conv((1, 1), inplanes => outplanes; stride, pad = SamePad(), bias = false),
                 norm_layer(outplanes))
end

# Downsample layer using max pooling
function downsample_pool(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm)
    pool = (stride == 1) ? identity : MeanPool((2, 2); stride, pad = SamePad())
    return Chain(pool,
                 Conv((1, 1), inplanes => outplanes; bias = false),
                 norm_layer(outplanes))
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

function downsample_block(downsample_fns, inplanes, planes, expansion; stride = 1,
                          norm_layer = BatchNorm)
    down_fn1, down_fn2 = downsample_fns
    if stride != 1 || inplanes != planes * expansion
        return down_fn1(inplanes, planes * expansion; stride, norm_layer)
    else
        return down_fn2(inplanes, planes * expansion; stride, norm_layer)
    end
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
function _make_downsample_fns(vec::Vector{<:Symbol}, layers)
    downs = []
    for i in vec
       @assert i in keys(shortcut_dict)
       "The shortcut type must be one of $(sort(collect(keys(shortcut_dict))))"
       push!(downs, shortcut_dict[i])
    end
    return downs
end
function _make_downsample_fns(sym::Symbol, layers)
    @assert sym in keys(shortcut_dict)
    "The shortcut type must be one of $(sort(collect(keys(shortcut_dict))))"
    return collect(shortcut_dict[sym] for _ in 1:length(layers))
end
_make_downsample_fns(vec::Vector{<:NTuple{2}}, layers) = vec
_make_downsample_fns(tup::NTuple{2}, layers) = collect(tup for _ in 1:length(layers))

# Stride for each block in the ResNet model
function get_stride(::AbstractResNetBlock, idxs::NTuple{2, Integer})
    return (idxs[1] == 1 || idxs[1] == 1) ? 2 : 1
end

# returns `DropBlock`s for each stage of the ResNet
function _drop_blocks(drop_block_rate::AbstractFloat)
    return [
        identity, identity,
        DropBlock(drop_block_rate, 5, 0.25), DropBlock(drop_block_rate, 3, 1.00)
    ]
end

function _make_layers(block::basicblock, norm_layer, stride)
    first_planes = block.planes ÷ block.reduction_factor
    outplanes = block.planes * expansion_factor(block)
    conv_bn1 = Chain(Conv((3, 3), block.inplanes => first_planes; stride, pad = 1, bias = false),
                     norm_layer(first_planes))
    conv_bn2 = Chain(Conv((3, 3), first_planes => outplanes; pad = 1, bias = false),
                     norm_layer(outplanes))
    layers = []
    push!(layers, conv_bn1, conv_bn2)
    return layers
end

function _make_layers(block::bottleneck, norm_layer, stride)
    width = fld(block.planes * block.base_width, 64) * block.cardinality
    first_planes = width ÷ block.reduction_factor
    outplanes = block.planes * expansion_factor(block)
    conv_bn1 = Chain(Conv((1, 1), block.inplanes => first_planes; bias = false),
                     norm_layer(first_planes))
    conv_bn2 = Chain(Conv((3, 3), first_planes => width; stride, pad = 1,
                          groups = block.cardinality, bias = false),
                     norm_layer(width))
    conv_bn3 = Chain(Conv((1, 1), width => outplanes; bias = false), norm_layer(outplanes))
    layers = []
    push!(layers, conv_bn1, conv_bn2, conv_bn3)
    return layers
end

function make_block(block::T, idxs::NTuple{2, Integer}; kwargs...) where {T <: AbstractResNetBlock}
    stage_idx, block_idx = idxs
    kwargs = Dict(kwargs)
    stride = get(kwargs, :stride_fn, get_stride)(block, idxs)
    expansion = expansion_factor(block)
    norm_layer = get(kwargs, :norm_layer, BatchNorm)
    layers = _make_layers(block, norm_layer, stride)
    activation = get(kwargs, :activation, relu)
    insert!(layers, 2, activation)
    if T <: bottleneck
        insert!(layers, 4, activation)
    end
    if haskey(kwargs, :drop_block_rate)
        layer_idx = T <: basicblock ? 2 : 3
        dropblock = _drop_blocks(kwargs[:drop_block_rate])[stage_idx]
        insert!(layers, layer_idx, dropblock)
    end
    if haskey(kwargs, :attn_fn)
        attn_layer = kwargs[:attn_fn](block.planes)
        push!(layers, attn_layer)
    end
    if haskey(kwargs, :drop_path_rate)
        droppath = DropPath(kwargs[:droppath_rates][block_idx])
        push!(layers, droppath)
    end
    if haskey(kwargs, :downsample_fns)
        downsample_tup = kwargs[:downsample_fns][stage_idx]
        downsample = downsample_block(downsample_tup, block.inplanes, block.planes, expansion; stride)
        connection = get(kwargs, :connection, addact)$activation
        return Parallel(connection, downsample, Chain(layers...))
    else
        return Chain(layers...)
    end
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
  - `replace_stem_pool`: Whether to replace the default 3x3 max pooling layer with a
    3x3 convolution with stride 2 and a normalisation layer.
  - `norm_layer`: The normalisation layer used in the stem.
  - `activation`: The activation function used in the stem.
"""
function resnet_stem(; stem_type::Symbol = :default, inchannels::Integer = 3,
                     replace_stem_pool::Bool = false, norm_layer = BatchNorm, activation = relu)
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

# Makes the main stages of the ResNet model. This is an internal function and should not be 
# used by end-users. `block_fn` is a function that returns a single block of the ResNet. 
# See `basicblock` and `bottleneck` for examples. A block must define a function 
# `expansion(::typeof(block))` that returns the expansion factor of the block.
function resnet_stages(block_type, channels, block_repeats, inplanes; kwargs...)
    stages = []
    kwargs = Dict(kwargs)
    cardinality = get(kwargs, :cardinality, 1)
    base_width = get(kwargs, :base_width, 64)
    reduction_factor = get(kwargs, :reduction_factor, 1)
    ## Construct each stage
    for (stage_idx, (planes, num_blocks)) in enumerate(zip(channels, block_repeats))
        ## Construct the blocks for each stage
        blocks = []
        for block_idx in 1:num_blocks
            block_struct = block_type(inplanes, planes, reduction_factor, base_width, cardinality)
            block = make_block(block_struct, (stage_idx, block_idx); kwargs...)
            inplanes = planes * expansion_factor(block_struct)
            push!(blocks, block)
        end
        push!(stages, Chain(blocks...))
    end
    return Chain(stages...), inplanes
end

function resnet(block_fn, layers, downsample_opt = :B; inchannels::Integer = 3,
                nclasses::Integer = 1000, stem = first(resnet_stem(; inchannels)),
                inplanes::Integer = 64, kwargs...)
    kwargs = Dict(kwargs)
    ## Feature Blocks
    channels = collect(64 * 2^i for i in range(0, length(layers)))
    downsample_fns = _make_downsample_fns(downsample_opt, layers)
    stage_blocks, num_features = resnet_stages(block_fn, channels, layers, inplanes; downsample_fns, kwargs...)
    ## Classifier head
    # num_features = 512 * expansion_factor(block_fn)
    pool_layer = get(kwargs, :pool_layer, AdaptiveMeanPool((1, 1)))
    use_conv = get(kwargs, :use_conv, false)
    # Pooling
    if pool_layer === identity
        @assert use_conv
        "Pooling can only be disabled if classifier is also removed or a convolution-based classifier is used"
    end
    flatten_in_pool = !use_conv && pool_layer !== identity
    global_pool = flatten_in_pool ? Chain(pool_layer, MLUtils.flatten) : pool_layer
    # Fully-connected layer
    fc = create_fc(num_features, nclasses; use_conv)
    classifier = Chain(global_pool, Dropout(get(kwargs, :dropout_rate, 0)), fc)
    return Chain(Chain(stem, stage_blocks), classifier)
end

# block-layer configurations for ResNet-like models
const resnet_config = Dict(18 => (basicblock, [2, 2, 2, 2]),
                           34 => (basicblock, [3, 4, 6, 3]),
                           50 => (bottleneck, [3, 4, 6, 3]),
                           101 => (bottleneck, [3, 4, 23, 3]),
                           152 => (bottleneck, [3, 8, 36, 3]))
