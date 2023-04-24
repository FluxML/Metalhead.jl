## ResNet blocks

"""
    basicblock(inplanes::Integer, planes::Integer; stride::Integer = 1,
               reduction_factor::Integer = 1, activation = relu,
               norm_layer = BatchNorm, revnorm::Bool = false,
               drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity)

Creates a basic residual block (see [reference](https://arxiv.org/abs/1512.03385v1)).
This function creates the layers. For more configuration options and to see the function
used to build the block for the model, see [`Metalhead.basicblock_builder`](@ref).

# Arguments

- `inplanes`: number of input feature maps
- `planes`: number of feature maps for the block
- `stride`: the stride of the block
- `reduction_factor`: the factor by which the input feature maps are reduced before
the first convolution.
- `activation`: the activation function to use.
- `norm_layer`: the normalization layer to use.
- `revnorm`: set to `true` to place the normalisation layer before the convolution
- `drop_block`: the drop block layer
- `drop_path`: the drop path layer
- `attn_fn`: the attention function to use. See [`squeeze_excite`](@ref) for an example.
"""
function basicblock(inplanes::Integer, 
                    planes::Integer; 
                    stride::Integer = 1,
                    reduction_factor::Integer = 1, 
                    activation = relu,
                    norm_layer = BatchNorm, 
                    revnorm::Bool = false,
                    drop_block = identity, 
                    drop_path = identity,
                    attn_fn = planes -> identity)
  
    first_planes = planes ÷ reduction_factor
    conv_bn1 = conv_norm((3, 3), inplanes, first_planes, identity; norm_layer, revnorm,
                         stride, pad = 1)
    conv_bn2 = conv_norm((3, 3), first_planes, planes, identity; norm_layer, revnorm,
                         pad = 1)
    layers = [conv_bn1..., drop_block, activation, conv_bn2..., attn_fn(planes),
        drop_path]
    return Chain(filter!(!=(identity), layers)...)
end

"""
    bottleneck(inplanes::Integer, planes::Integer; stride::Integer,
               cardinality::Integer = 1, base_width::Integer = 64,
               reduction_factor::Integer = 1, activation = relu,
               norm_layer = BatchNorm, revnorm::Bool = false,
               drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity)

Creates a bottleneck residual block (see [reference](https://arxiv.org/abs/1512.03385v1)).
This function creates the layers. For more configuration options and to see the function
used to build the block for the model, see [`Metalhead.bottleneck_builder`](@ref).

# Arguments

- `inplanes`: number of input feature maps
- `planes`: number of feature maps for the block
- `stride`: the stride of the block
- `cardinality`: the number of groups in the convolution.
- `base_width`: the number of output feature maps for each convolutional group.
- `reduction_factor`: the factor by which the input feature maps are reduced before the first
  convolution.
- `activation`: the activation function to use.
- `norm_layer`: the normalization layer to use.
- `revnorm`: set to `true` to place the normalisation layer before the convolution
- `drop_block`: the drop block layer
- `drop_path`: the drop path layer
- `attn_fn`: the attention function to use. See [`squeeze_excite`](@ref) for an example.
"""
function bottleneck(inplanes::Integer, planes::Integer; stride::Integer,
                    cardinality::Integer = 1, base_width::Integer = 64,
                    reduction_factor::Integer = 1, activation = relu,
                    norm_layer = BatchNorm, revnorm::Bool = false,
                    drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity)
    width = fld(planes * base_width, 64) * cardinality
    first_planes = width ÷ reduction_factor
    outplanes = planes * 4
    conv_bn1 = conv_norm((1, 1), inplanes, first_planes, activation; norm_layer,
                         revnorm)
    conv_bn2 = conv_norm((3, 3), first_planes, width, identity; norm_layer, revnorm,
                         stride, pad = 1, groups = cardinality)
    conv_bn3 = conv_norm((1, 1), width, outplanes, identity; norm_layer, revnorm)
    layers = [conv_bn1..., conv_bn2..., drop_block, activation, conv_bn3...,
        attn_fn(outplanes), drop_path]
    return Chain(filter!(!=(identity), layers)...)
end

"""
    bottle2neck(inplanes::Integer, planes::Integer; stride::Integer = 1,
                cardinality::Integer = 1, base_width::Integer = 26,
                scale::Integer = 4, activation = relu, norm_layer = BatchNorm,
                revnorm::Bool = false, attn_fn = planes -> identity)

Creates a bottleneck block as described in the Res2Net paper. ([reference](https://arxiv.org/abs/1904.01169))
This function creates the layers. For more configuration options and to see the function
used to build the block for the model, see [`Metalhead.bottle2neck_builder`](@ref).

# Arguments

- `inplanes`: number of input feature maps
- `planes`: number of feature maps for the block
- `stride`: the stride of the block
- `cardinality`: the number of groups in the 3x3 convolutions.
- `base_width`: the number of output feature maps for each convolutional group.
- `scale`: the number of feature groups in the block. See the [paper](https://arxiv.org/abs/1904.01169)
  for more details.
- `activation`: the activation function to use.
- `norm_layer`: the normalization layer to use.
- `revnorm`: set to `true` to place the batch norm before the convolution
- `attn_fn`: the attention function to use. See [`squeeze_excite`](@ref) for an example.
"""
function bottle2neck(inplanes::Integer, planes::Integer; stride::Integer = 1,
                     cardinality::Integer = 1, base_width::Integer = 26,
                     scale::Integer = 4, activation = relu, is_first::Bool = false,
                     norm_layer = BatchNorm, revnorm::Bool = false,
                     attn_fn = planes -> identity)
    width = fld(planes * base_width, 64) * cardinality
    outplanes = planes * 4
    pool = is_first && scale > 1 ? MeanPool((3, 3); stride, pad = 1) : identity
    conv_bns = [Chain(conv_norm((3, 3), width, width, activation; norm_layer, stride,
                                pad = 1, groups = cardinality)...)
                for _ in 1:max(1, scale - 1)]
    reslayer = is_first ? Parallel(cat_channels, pool, conv_bns...) :
               Parallel(cat_channels, identity, Chain(PairwiseFusion(+, conv_bns...)))
    tuplify = is_first ? x -> tuple(x...) : x -> tuple(x[1], tuple(x[2:end]...))
    layers = [
        conv_norm((1, 1), inplanes, width * scale, activation;
                  norm_layer, revnorm)...,
        chunk$(; size = width, dims = 3), tuplify, reslayer,
        conv_norm((1, 1), width * scale, outplanes, activation;
                  norm_layer, revnorm)...,
        attn_fn(outplanes),
    ]
    return Chain(filter!(!=(identity), layers)...)
end

## Downsample layers

"""
    downsample_conv(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                    norm_layer = BatchNorm, revnorm::Bool = false)

Creates a 1x1 convolutional downsample layer as used in ResNet.

# Arguments

- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `stride`: the stride of the convolution
- `norm_layer`: the normalization layer to use.
- `revnorm`: set to `true` to place the normalisation layer before the convolution
"""
function downsample_conv(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm, revnorm::Bool = false)
    return Chain(conv_norm((1, 1), inplanes, outplanes, identity; norm_layer, revnorm,
                           pad = SamePad(), stride)...)
end

"""
    downsample_pool(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                    norm_layer = BatchNorm, revnorm::Bool = false)

Creates a pooling-based downsample layer as described in the
[Bag of Tricks](https://arxiv.org/abs/1812.01187v1) paper. This adds an average pooling layer
of size `(2, 2)` with `stride` followed by a 1x1 convolution.

# Arguments

- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `stride`: the stride of the convolution
- `norm_layer`: the normalization layer to use.
- `revnorm`: set to `true` to place the normalisation layer before the convolution
"""
function downsample_pool(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm, revnorm::Bool = false)
    pool = stride == 1 ? identity : MeanPool((2, 2); stride, pad = SamePad())
    return Chain(pool,
                 conv_norm((1, 1), inplanes, outplanes, identity; norm_layer,
                           revnorm)...)
end

# TODO - figure out how to make this work when outplanes < inplanes
"""
    downsample_identity(inplanes::Integer, outplanes::Integer; kwargs...)

Creates an identity downsample layer. This returns `identity` if `inplanes == outplanes`.
If `outplanes > inplanes`, it maps the input to `outplanes` channels using a 1x1 max pooling
layer and zero padding.

!!! warning
    
    This does not currently support the scenario where `inplanes > outplanes`.

# Arguments

- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps

Note that kwargs are ignored and only included for compatibility with other downsample layers.
"""
function downsample_identity(inplanes::Integer, outplanes::Integer; kwargs...)
    if outplanes > inplanes
        return Chain(MaxPool((1, 1); stride = 2),
                     y -> cat_channels(y,
                                       zeros(eltype(y), size(y, 1), size(y, 2),
                                             outplanes - inplanes, size(y, 4))))
    else
        return identity
    end
end

# Shortcut configurations for the ResNet variants
const RESNET_SHORTCUTS = Dict(:A => (downsample_identity, downsample_identity),
                              :B => (downsample_conv, downsample_identity),
                              :C => (downsample_conv, downsample_conv),
                              :D => (downsample_pool, downsample_identity))

# returns `DropBlock`s for each stage of the ResNet as in timm.
# TODO - add experimental options for DropBlock as part of the API (#188)
# function _drop_blocks(dropblock_prob::AbstractFloat)
#     return [
#         identity, identity,
#         DropBlock(dropblock_prob, 5, 0.25), DropBlock(dropblock_prob, 3, 1.00),
#     ]
# end

"""
    resnet_stem(; stem_type = :default, inchannels::Integer = 3, replace_stem_pool = false,
                  norm_layer = BatchNorm, activation = relu)

Builds a stem to be used in a ResNet model. See the `stem` argument of [`resnet`](@ref) for details
on how to use this function.

# Arguments

- `stem_type`: The type of stem to be built. One of `[:default, :deep, :deep_tiered]`.
  
    + `:default`: Builds a stem based on the default ResNet stem, which consists of a single
      7x7 convolution with stride 2 and a normalisation layer followed by a 3x3 max pooling
      layer with stride 2.
    + `:deep`: This borrows ideas from other papers ([InceptionResNetv2](https://arxiv.org/abs/1602.07261),
      for example) in using a deeper stem with 3 successive 3x3 convolutions having normalisation
      layers after each one. This is followed by a 3x3 max pooling layer with stride 2.
    + `:deep_tiered`: A variant of the `:deep` stem that has a larger width in the second
      convolution. This is an experimental variant from the `timm` library in Python that
      shows peformance improvements over the `:deep` stem in some cases.

- `inchannels`: number of input channels
- `replace_pool`: Set to true to replace the max pooling layers with a 3x3 convolution +
  normalization with a stride of two.
- `norm_layer`: The normalisation layer used in the stem.
- `activation`: The activation function used in the stem.
"""
function resnet_stem(stem_type::Symbol = :default; inchannels::Integer = 3,
                     replace_pool::Bool = false, activation = relu,
                     norm_layer = BatchNorm, revnorm::Bool = false)
    # Check for valid stem types
    deep_stem = if stem_type === :deep || stem_type === :deep_tiered
        true
    elseif stem_type === :default
        false
    else
        throw(ArgumentError("Unsupported stem type $stem_type. Must be one of 
                             [:default, :deep, :deep_tiered]"))
    end
    # Main stem
    inplanes = deep_stem ? stem_width * 2 : 64
    # Deep stem that uses three successive 3x3 convolutions instead of a single 7x7 convolution
    if deep_stem
        if stem_type === :deep
            stem_channels = (stem_width, stem_width)
        elseif stem_type === :deep_tiered
            stem_channels = (3 * (stem_width ÷ 4), stem_width)
        end
        conv1 = Chain(conv_norm((3, 3), inchannels, stem_channels[1], activation;
                                norm_layer, revnorm, stride = 2, pad = 1)...,
                      conv_norm((3, 3), stem_channels[1], stem_channels[2], activation;
                                norm_layer, pad = 1)...,
                      Conv((3, 3), stem_channels[2] => inplanes; pad = 1, bias = false))
    else
        conv1 = Conv((7, 7), inchannels => inplanes; stride = 2, pad = 3, bias = false)
    end
    bn1 = norm_layer(inplanes, activation)
    # Stem pooling
    stempool = replace_pool ?
               Chain(conv_norm((3, 3), inplanes, inplanes, activation; norm_layer,
                               revnorm, stride = 2, pad = 1)...) :
               MaxPool((3, 3); stride = 2, pad = 1)
    return Chain(conv1, bn1, stempool)
end

# Callbacks for channel and stride calculations for each block in a ResNet

"""
    resnet_planes(block_repeats::AbstractVector{<:Integer})

Default callback for determining the number of channels in each block in a ResNet model.

# Arguments

`block_repeats`: A `Vector` of integers specifying the number of times each block is repeated
in each stage of the ResNet model. For example, `[3, 4, 6, 3]` is the configuration used in
ResNet-50, which has 3 blocks in the first stage, 4 blocks in the second stage, 6 blocks in the
third stage and 3 blocks in the fourth stage.
"""
function resnet_planes(block_repeats::AbstractVector{<:Integer})
    return collect(Iterators.flatten((64 * 2^(stage_idx - 1) for _ in 1:stages)
                                     for (stage_idx, stages) in enumerate(block_repeats)))
end

"""
    resnet_stride(stage_idx::Integer, block_idx::Integer)

Default callback for determining the stride of a block in a ResNet model.
Returns `2` for the first block in every stage except the first stage and `1` for all other
blocks.

# Arguments

  - `stage_idx`: The index of the stage in the ResNet model.
  - `block_idx`: The index of the block in the stage.
"""
function resnet_stride(stage_idx::Integer, block_idx::Integer)
    return stage_idx == 1 || block_idx != 1 ? 1 : 2
end

"""
    resnet(block_type, block_repeats::AbstractVector{<:Integer},
           downsample_opt::NTuple{2, Any} = (downsample_conv, downsample_identity);
           cardinality::Integer = 1, base_width::Integer = 64,
           inplanes::Integer = 64, reduction_factor::Integer = 1,
           connection = addact, activation = relu,
           norm_layer = BatchNorm, revnorm::Bool = false,
           attn_fn = planes -> identity, pool_layer = AdaptiveMeanPool((1, 1)),
           use_conv::Bool = false, dropblock_prob = nothing,
           stochastic_depth_prob = nothing, dropout_prob = nothing,
           imsize::Dims{2} = (256, 256), inchannels::Integer = 3,
           nclasses::Integer = 1000, kwargs...)

Creates a generic ResNet-like model that is used to create The higher-level model constructors like ResNet,
Wide ResNet, ResNeXt and Res2Net. For an _even_ more generic model API, see [`Metalhead.build_resnet`](@ref).

# Arguments

- `block_type`: The type of block to be used in the model. This can be one of [`Metalhead.basicblock`](@ref),
  [`Metalhead.bottleneck`](@ref) and [`Metalhead.bottle2neck`](@ref). `basicblock` is used in the
  original ResNet paper for ResNet-18 and ResNet-34, and `bottleneck` is used in the original ResNet-50
  and ResNet-101 models, as well as for the Wide ResNet and ResNeXt models. `bottle2neck` is introduced in
  the `Res2Net` paper.
- `block_repeats`: A `Vector` of integers specifying the number of times each block is repeated
  in each stage of the ResNet model. For example, `[3, 4, 6, 3]` is the configuration used in
  ResNet-50, which has 3 blocks in the first stage, 4 blocks in the second stage, 6 blocks in the
  third stage and 3 blocks in the fourth stage.
- `downsample_opt`: A `NTuple` of two callbacks that are used to determine the downsampling
  operation to be used in the model. The first callback is used to determine the convolutional
  operation to be used in the downsampling operation and the second callback is used to determine
  the identity operation to be used in the downsampling operation.
- `cardinality`: The number of groups to be used in the 3x3 convolutional layer in the bottleneck
  block. This is usually modified from the default value of `1` in the ResNet models to `32` or `64`
  in the `ResNeXt` models.
- `base_width`: The base width of the convolutional layer in the blocks of the model.
- `inplanes`: The number of input channels in the first convolutional layer.
- `reduction_factor`: The reduction factor used in the model.
- `connection`: This is a function that determines the residual connection in the model. For
  `resnets`, either of [`Metalhead.addact`](@ref) or [`Metalhead.actadd`](@ref) is recommended.
- `norm_layer`: The normalisation layer to be used in the model.
- `revnorm`: set to `true` to place the normalisation layers before the convolutions
- `attn_fn`: A callback that is used to determine the attention function to be used in the model.
  See [`Metalhead.Layers.squeeze_excite`](@ref) for an example.
- `pool_layer`: A fully-instantiated pooling layer passed in to be used by the classifier head.
  For example, `AdaptiveMeanPool((1, 1))` is used in the ResNet family by default, but something
  like `MeanPool((3, 3))` should also work provided the dimensions after applying the pooling
  layer are compatible with the rest of the classifier head.
- `use_conv`: Set to true to use convolutions instead of identity operations in the model.
- `dropblock_prob`: `DropBlock` probability to be used in the model. Set to `nothing` to disable
  DropBlock. See [`Metalhead.DropBlock`](@ref) for more details.
- `stochastic_depth_prob`: `StochasticDepth` probability to be used in the model. Set to `nothing`
  to disable StochasticDepth. See [`Metalhead.StochasticDepth`](@ref) for more details.
- `dropout_prob`: `Dropout` probability to be used in the classifier head. Set to `nothing` to
  disable Dropout.
- `imsize`: The size of the input (height, width).
- `inchannels`: The number of input channels.
- `nclasses`: The number of output classes.
- `kwargs`: Additional keyword arguments to be passed to the block builder (note: ignore this
  argument if you are not sure what it does. To know more about how this works, check out the
  section of the documentation that talks about builders in Metalhead and specifically for the
  ResNet block functions).
"""
function resnet(block_type, block_repeats::AbstractVector{<:Integer},
                downsample_opt::NTuple{2, Any} = (downsample_conv, downsample_identity);
                cardinality::Integer = 1, base_width::Integer = 64,
                inplanes::Integer = 64, reduction_factor::Integer = 1,
                connection = addact, activation = relu,
                norm_layer = BatchNorm, revnorm::Bool = false,
                attn_fn = planes -> identity, pool_layer = AdaptiveMeanPool((1, 1)),
                use_conv::Bool = false, dropblock_prob = nothing,
                stochastic_depth_prob = nothing, dropout_prob = nothing,
                imsize::Dims{2} = (256, 256), inchannels::Integer = 3,
                nclasses::Integer = 1000, kwargs...)
    # Build stem
    stem = resnet_stem(; inchannels)
    # Block builder
    if block_type == basicblock
        @assert cardinality==1 "Cardinality must be 1 for `basicblock`"
        @assert base_width==64 "Base width must be 64 for `basicblock`"
        get_layers = basicblock_builder(block_repeats; inplanes, reduction_factor,
                                        activation, norm_layer, revnorm, attn_fn,
                                        dropblock_prob, stochastic_depth_prob,
                                        stride_fn = resnet_stride,
                                        planes_fn = resnet_planes,
                                        downsample_tuple = downsample_opt, kwargs...)
    elseif block_type == bottleneck
        get_layers = bottleneck_builder(block_repeats; inplanes, cardinality, base_width,
                                        reduction_factor, activation, norm_layer, revnorm,
                                        attn_fn, dropblock_prob, stochastic_depth_prob,
                                        stride_fn = resnet_stride,
                                        planes_fn = resnet_planes,
                                        downsample_tuple = downsample_opt, kwargs...)
    elseif block_type == bottle2neck
        @assert isnothing(dropblock_prob) "DropBlock not supported for `bottle2neck`.
        Set `dropblock_prob` to nothing."
        @assert isnothing(stochastic_depth_prob) "StochasticDepth not supported for `bottle2neck`.
        Set `stochastic_depth_prob` to nothing."
        @assert reduction_factor==1 "Reduction factor not supported for `bottle2neck`.
        Set `reduction_factor` to 1."
        get_layers = bottle2neck_builder(block_repeats; inplanes, cardinality, base_width,
                                         activation, norm_layer, revnorm, attn_fn,
                                         stride_fn = resnet_stride,
                                         planes_fn = resnet_planes,
                                         downsample_tuple = downsample_opt, kwargs...)
    else
        # TODO: write better message when we have link to dev docs for resnet
        throw(ArgumentError("Unknown block type $block_type"))
    end
    classifier_fn = nfeatures -> create_classifier(nfeatures, nclasses; dropout_prob,
                                                   pool_layer, use_conv)
    return build_resnet((imsize..., inchannels), stem, get_layers, block_repeats,
                        connection$activation, classifier_fn)
end

function resnet(block_fn, block_repeats, downsample_opt::Symbol = :B; kwargs...)
    return resnet(block_fn, block_repeats, RESNET_SHORTCUTS[downsample_opt]; kwargs...)
end

# block-layer configurations for ResNet-like models
const RESNET_CONFIGS = Dict(18 => (basicblock, [2, 2, 2, 2]),
                            34 => (basicblock, [3, 4, 6, 3]),
                            50 => (bottleneck, [3, 4, 6, 3]),
                            101 => (bottleneck, [3, 4, 23, 3]),
                            152 => (bottleneck, [3, 8, 36, 3]))
# block configurations for larger ResNet-like models that do not use
# depths 18 and 34
const LRESNET_CONFIGS = Dict(50 => (bottleneck, [3, 4, 6, 3]),
                             101 => (bottleneck, [3, 4, 23, 3]),
                             152 => (bottleneck, [3, 8, 36, 3]))
