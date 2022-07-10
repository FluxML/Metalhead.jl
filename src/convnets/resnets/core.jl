## It is recommended to check out the user guide for more information.

### ResNet blocks
## These functions return a block to be used inside of a ResNet model.
## The individual arguments are explained in the documentation of the functions.
## Note that for these blocks to be used by the `_make_blocks` function, they must define
## a dispatch `expansion(::typeof(fn))` that returns the expansion factor of the block 
## (i.e. the multiplicative factor by which the number of channels in the input is increased).
## The `_make_blocks` function will then call the `expansion` function to determine the
## expansion factor of each block and use this to construct the stages of the model.

"""
    basicblock(inplanes, planes; stride = 1, downsample = identity,
               reduction_factor = 1, dilation = 1, first_dilation = dilation,
               activation = relu, connection = addact\$activation,
               norm_layer = BatchNorm, drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity, attn_args::NamedTuple = NamedTuple())

Creates a basic ResNet block.

# Arguments

  - `inplanes`: number of input feature maps
  - `planes`: number of feature maps for the block
  - `stride`: the stride of the block
  - `downsample`: the downsampling function to use
  - `reduction_factor`: the reduction factor that the input feature maps are reduced by before the first
    convolution.
  - `dilation`: the dilation of the second convolution.
  - `first_dilation`: the dilation of the first convolution.
  - `activation`: the activation function to use.
  - `connection`: the function applied to the output of residual and skip paths in
    a block. See [`addact`](#) and [`actadd`](#) for an example. Note that this uses
    PartialFunctions.jl to pass in the activation function with the notation `addact\$activation`.
  - `norm_layer`: the normalization layer to use.
  - `drop_block`: the drop block layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `drop_path`: the drop path layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
  - `attn_args`: a NamedTuple that contains none, some or all of the arguments to be passed to the
    attention function.
"""
function basicblock(inplanes, planes; stride = 1, downsample = identity,
                    reduction_factor = 1, dilation = 1, first_dilation = dilation,
                    activation = relu, connection = addact$activation,
                    norm_layer = BatchNorm, drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity, attn_args::NamedTuple = NamedTuple())
    expansion = expansion_factor(basicblock)
    first_planes = planes ÷ reduction_factor
    outplanes = planes * expansion
    conv_bn1 = Chain(Conv((3, 3), inplanes => first_planes; stride, pad = first_dilation,
                          dilation = first_dilation, bias = false),
                     norm_layer(first_planes))
    drop_block = drop_block
    conv_bn2 = Chain(Conv((3, 3), first_planes => outplanes; pad = dilation,
                          dilation = dilation, bias = false),
                     norm_layer(outplanes))
    attn_layer = attn_fn(outplanes; attn_args...)
    return Parallel(connection, downsample,
                    Chain(conv_bn1, drop_block, activation, conv_bn2, attn_layer,
                          drop_path))
end
expansion_factor(::typeof(basicblock)) = 1

"""
    bottleneck(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
               base_width = 64, reduction_factor = 1, first_dilation = 1,
               activation = relu, connection = addact\$activation,
               norm_layer = BatchNorm, drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity, attn_args::NamedTuple = NamedTuple())

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
  - `first_dilation`: the dilation of the 3x3 convolution.
  - `activation`: the activation function to use.
  - `connection`: the function applied to the output of residual and skip paths in
    a block. See [`addact`](#) and [`actadd`](#) for an example. Note that this uses
    PartialFunctions.jl to pass in the activation function with the notation `addact\$activation`.
  - `norm_layer`: the normalization layer to use.
  - `drop_block`: the drop block layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `drop_path`: the drop path layer. This is usually initialised in the `_make_blocks`
    function and passed in.
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
  - `attn_args`: a NamedTuple that contains none, some or all of the arguments to be passed to the
    attention function.
"""
function bottleneck(inplanes, planes; stride = 1, downsample = identity, cardinality = 1,
                    base_width = 64, reduction_factor = 1, first_dilation = 1,
                    activation = relu, connection = addact$activation,
                    norm_layer = BatchNorm, drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity, attn_args::NamedTuple = NamedTuple())
    expansion = expansion_factor(bottleneck)
    width = floor(Int, planes * (base_width / 64)) * cardinality
    first_planes = width ÷ reduction_factor
    outplanes = planes * expansion
    conv_bn1 = Chain(Conv((1, 1), inplanes => first_planes; bias = false),
                     norm_layer(first_planes, activation))
    conv_bn2 = Chain(Conv((3, 3), first_planes => width; stride, pad = first_dilation,
                          dilation = first_dilation, groups = cardinality, bias = false),
                     norm_layer(width))
    conv_bn3 = Chain(Conv((1, 1), width => outplanes; bias = false), norm_layer(outplanes))
    attn_layer = attn_fn(outplanes; attn_args...)
    return Parallel(connection, downsample,
                    Chain(conv_bn1, conv_bn2, drop_block, activation, conv_bn3,
                          attn_layer, drop_path))
end
expansion_factor(::typeof(bottleneck)) = 4

"""
    resnet_stem(; stem_type = :default, inchannels = 3, replace_stem_pool = false,
                  norm_layer = BatchNorm, activation = relu)

Builds a stem to be used in a ResNet model. See the `stem` argument of [`resnet`](#) for details
on how to use this function.

# Arguments

  - `stem_type`: The type of stem to be built. One of `[:default, :deep, :deep_tiered]`.
    
      + `:default`: Builds a stem based on the default ResNet stem, which consists of a single
        7x7 convolution with stride 2 and a normalisation layer followed by a 3x3
        max pooling layer with stride 2.
      + `:deep`: This borrows ideas from other papers (InceptionResNet-v2, for example) in using a
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

### Downsampling layers
## These will almost never be used directly. They are used by the `_make_blocks` function to 
## build the downsampling layers. In most cases, these defaults will not need to be changed. 
## If you wish to write your own ResNet model using the `_make_blocks` function, you can use 
## this function to build the downsampling layers.

# Downsample layer using convolutions.
function downsample_conv(kernel_size, inplanes, outplanes; stride = 1, dilation = 1,
                         norm_layer = BatchNorm)
    kernel_size = stride == 1 && dilation == 1 ? (1, 1) : kernel_size
    dilation = kernel_size[1] > 1 ? dilation : 1
    pad = ((stride - 1) + dilation * (kernel_size[1] - 1)) ÷ 2
    return Chain(Conv(kernel_size, inplanes => outplanes; stride, pad,
                      dilation, bias = false),
                 norm_layer(outplanes))
end

# Downsample layer using max pooling
function downsample_pool(kernel_size, inplanes, outplanes; stride = 1, dilation = 1,
                         norm_layer = BatchNorm)
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

# Downsample layer which is an identity projection. Uses max pooling
# when the output size is more than the input size.
function downsample_identity(kernel_size, inplanes, outplanes; kwargs...)
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

"""
    downsample_block(downsample_fn, inplanes, planes, expansion; kernel_size = (1, 1),
                     stride = 1, dilation = 1, norm_layer = BatchNorm)

Wrapper function that makes it easier to build a downsample block inside a ResNet model.
This function is almost never used directly or customised by the user.

# Arguments

  - `downsample_fn`: The function to use for downsampling in skip connections. Recommended usage
    is passing in either `downsample_conv` or `downsample_pool`.
  - `inplanes`: The number of input feature maps.
  - `planes`: The number of output feature maps.
  - `expansion`: The expansion factor of the block.
  - `kernel_size`: The size of the convolutional kernel.
  - `stride`: The stride of the convolutional layer.
  - `dilation`: The dilation of the convolutional layer.
  - `norm_layer`: The normalisation layer to be used.
"""
function downsample_block(downsample_fns, inplanes, planes, expansion;
                          kernel_size = (1, 1), stride = 1, dilation = 1,
                          norm_layer = BatchNorm)
    down_fn1, down_fn2 = downsample_fns
    if stride != 1 || inplanes != planes * expansion
        downsample = down_fn2(kernel_size, inplanes, planes * expansion;
                              stride, dilation, norm_layer)
    else
        downsample = down_fn1(kernel_size, inplanes, planes * expansion;
                              stride, dilation, norm_layer)
    end
    return downsample
end

const shortcut_dict = Dict(:A => (downsample_identity, downsample_identity),
                           :B => (downsample_conv, downsample_identity),
                           :C => (downsample_conv, downsample_conv))

function _make_downsample_fns(vec::Vector{T}) where {T}
    if T <: Symbol
        downs = []
        for i in vec
            @assert i in keys(shortcut_dict)
            "The shortcut type must be one of $(sort(collect(keys(shortcut_dict))))"
            push!(downs, shortcut_dict[i])
        end
        return downs
    elseif T <: NTuple{2}
        return vec
    else
        throw(ArgumentError("The shortcut list must be a `Vector` of `Symbol`s or `NTuple{2}`s"))
    end
end

# Makes the main stages of the ResNet model. This is an internal function and should not be 
# used by end-users. `block_fn` is a function that returns a single block of the ResNet. 
# See `basicblock` and `bottleneck` for examples. A block must define a function 
# `expansion(::typeof(block))` that returns the expansion factor of the block.
function _make_blocks(block_fn, channels, block_repeats, inplanes; output_stride = 32,
                      downsample_fns::Vector, drop_rates::NamedTuple,
                      block_args::NamedTuple)
    @assert output_stride in (8, 16, 32) "Invalid `output_stride`. Must be one of (8, 16, 32)"
    expansion = expansion_factor(block_fn)
    stages = []
    net_block_idx = 1
    net_stride = 4
    dilation = prev_dilation = 1
    # Stochastic depth linear decay rule (DropPath)
    dp_rates = LinRange{Float32}(0.0, get(drop_rates, :drop_path_rate, 0),
                                 sum(block_repeats))
    # DropBlock rate
    dbr = get(drop_rates, :drop_block_rate, 0)
    ## Construct each stage
    for (stage_idx, itr) in enumerate(zip(channels, block_repeats, _drop_blocks(dbr),
                                          downsample_fns))
        # Number of planes in each stage, number of blocks in each stage, and the drop block rate
        planes, num_blocks, drop_block, down_fns = itr
        # Stride calculations for each stage
        stride = stage_idx == 1 ? 1 : 2
        if net_stride >= output_stride
            dilation *= stride
            stride = 1
        else
            net_stride *= stride
        end
        # Downsample block; either a (default) convolution-based block or a pooling-based block
        downsample = downsample_block(down_fns, inplanes, planes, expansion;
                                      stride, dilation)
        ## Construct the blocks for each stage
        blocks = []
        for block_idx in 1:num_blocks
            # Different behaviour for the first block of each stage
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

"""
    resnet(block_fn, layers; inchannels = 3, nclasses = 1000, output_stride = 32,
           stem = first(resnet_stem(; inchannels)), inplanes = 64,
           downsample_fn = downsample_conv, block_args::NamedTuple = NamedTuple(),
           drop_rates::NamedTuple = (dropout_rate = 0.0, drop_path_rate = 0.0,
                                     drop_block_rate = 0.0),
           classifier_args::NamedTuple = (pool_layer = AdaptiveMeanPool((1, 1)),
                                          use_conv = false))

This function creates the layers for many ResNet-like models. See the user guide for more
information.

!!! note
    
    If you are an end-user trying to use ResNet-like models, you should consider [`ResNet`](#)
    and similar higher-level functions instead. This version is significantly more customisable
    at the cost of being more complicated.

# Arguments

  - `block_fn`: The type of block to use inside the ResNet model. Must be either `:basicblock`,
    which is the standard ResNet block, or `:bottleneck`, which is the ResNet block with a
    bottleneck structure. See the [paper](https://arxiv.org/abs/1512.03385) for more details.

  - `layers`: A list of integers specifying the number of blocks in each stage. For example,
    `[3, 4, 6, 3]` would mean that the network would have 4 stages, with 3, 4, 6 and 3 blocks in
    each.
  - `nclasses`: The number of output classes.
  - `inchannels`: The number of input channels.
  - `output_stride`: The total stride of the network i.e. the amount by which the input is
    downsampled throughout the network. This is used to determine the output size from the
    backbone of the network. Must be one of `[8, 16, 32]`.
  - `stem`: A constructed ResNet stem, passed in to be used in the model. `inplanes` should be
    set to the number of output channels from this stem. Metalhead provides an in-built
    function for creating a stem (see [`resnet_stem`](#)) but you can also create your
    own (although this is not usually necessary).
  - `inplanes`: The number of output channels from the stem.
  - `downsample_type`: The type of downsampling to use. Either `:conv` or `:pool`. The former
    uses a traditional convolution-based downsampling, while the latter is an
    average-pooling-based downsampling that was suggested in the [Bag of Tricks](https://arxiv.org/abs/1812.01187)
    paper.
  - `block_args`: A `NamedTuple` that may define none, some or all the arguments to be passed
    to the block function. For more information regarding valid arguments, see
    the documentation for the block functions ([`basicblock`](#), [`bottleneck`](#)).
  - `drop_rates`: A `NamedTuple` that may define none, some or all of the following:
    
      + `dropout_rate`: The rate of dropout to be used in the classifier head.
      + `drop_path_rate`: Stochastic depth implemented using [`DropPath`](#).
      + `drop_block_rate`: `DropBlock` regularisation implemented using [`DropBlock`](#).
  - `classifier_args`: A `NamedTuple` that **must** specify the following arguments:
    
      + `pool_layer`: The adaptive pooling layer to use in the classifier head.
      + `use_conv`: Whether to use a 1x1 convolutional layer in the classifier head instead of a
        `Dense` layer.
"""
function resnet(block_fn, layers,
                downsample_list::Vector = collect(:B for _ in 1:length(layers));
                inchannels = 3, nclasses = 1000, output_stride = 32,
                stem = first(resnet_stem(; inchannels)), inplanes = 64,
                block_args::NamedTuple = NamedTuple(),
                drop_rates::NamedTuple = (dropout_rate = 0.0, drop_path_rate = 0.0,
                                          drop_block_rate = 0.0),
                classifier_args::NamedTuple = (pool_layer = AdaptiveMeanPool((1, 1)),
                                               use_conv = false))
    ## Feature Blocks
    channels = collect(64 * 2^i for i in range(0, length(layers)))
    downsample_fns = _make_downsample_fns(downsample_list)
    stage_blocks = _make_blocks(block_fn, channels, layers, inplanes;
                                output_stride, downsample_fns, drop_rates, block_args)
    ## Classifier head
    expansion = expansion_factor(block_fn)
    num_features = 512 * expansion
    pool_layer, use_conv = classifier_args
    # Pooling
    if pool_layer === identity
        @assert use_conv
        "Pooling can only be disabled if classifier is also removed or a convolution-based classifier is used"
    end
    flatten_in_pool = !use_conv && pool_layer !== identity
    global_pool = flatten_in_pool ? Chain(pool_layer, MLUtils.flatten) : pool_layer
    # Fully-connected layer
    fc = create_fc(num_features, nclasses; use_conv)
    classifier = Chain(global_pool, Dropout(get(drop_rates, :dropout_rate, 0)), fc)
    return Chain(Chain(stem, stage_blocks), classifier)
end

# block-layer configurations for ResNet-like models
const resnet_config = Dict(18 => (basicblock, [2, 2, 2, 2]),
                           34 => (basicblock, [3, 4, 6, 3]),
                           50 => (bottleneck, [3, 4, 6, 3]),
                           101 => (bottleneck, [3, 4, 23, 3]),
                           152 => (bottleneck, [3, 8, 36, 3]))
