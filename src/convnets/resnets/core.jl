"""
    basicblock(inplanes::Integer, planes::Integer; stride::Integer = 1,
               reduction_factor::Integer = 1, activation = relu,
               norm_layer = BatchNorm, revnorm::Bool = false,
               drop_block = identity, drop_path = identity,
               attn_fn = planes -> identity)

Creates a basic residual block (see [reference](https://arxiv.org/abs/1512.03385v1)).

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
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
"""
function basicblock(inplanes::Integer, planes::Integer; stride::Integer = 1,
                    reduction_factor::Integer = 1, activation = relu,
                    norm_layer = BatchNorm, revnorm::Bool = false,
                    drop_block = identity, drop_path = identity,
                    attn_fn = planes -> identity)
    first_planes = planes ÷ reduction_factor
    conv_bn1 = conv_norm((3, 3), inplanes => first_planes, identity; norm_layer, revnorm,
                         stride, pad = 1)
    conv_bn2 = conv_norm((3, 3), first_planes => planes, identity; norm_layer, revnorm,
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
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
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
    conv_bn1 = conv_norm((1, 1), inplanes => first_planes, activation; norm_layer,
                         revnorm)
    conv_bn2 = conv_norm((3, 3), first_planes => width, identity; norm_layer, revnorm,
                         stride, pad = 1, groups = cardinality)
    conv_bn3 = conv_norm((1, 1), width => outplanes, identity; norm_layer, revnorm)
    layers = [conv_bn1..., conv_bn2..., drop_block, activation, conv_bn3...,
        attn_fn(outplanes), drop_path]
    return Chain(filter!(!=(identity), layers)...)
end

# Downsample layer using convolutions.
function downsample_conv(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm, revnorm::Bool = false)
    return Chain(conv_norm((1, 1), inplanes => outplanes, identity; norm_layer, revnorm,
                           pad = SamePad(), stride)...)
end

# Downsample layer using max pooling
function downsample_pool(inplanes::Integer, outplanes::Integer; stride::Integer = 1,
                         norm_layer = BatchNorm, revnorm::Bool = false)
    pool = stride == 1 ? identity : MeanPool((2, 2); stride, pad = SamePad())
    return Chain(pool,
                 conv_norm((1, 1), inplanes => outplanes, identity; norm_layer,
                           revnorm)...)
end

# Downsample layer which is an identity projection. Uses max pooling
# when the output size is more than the input size.
# TODO - figure out how to make this work when outplanes < inplanes
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

# Shortcut configurations for the ResNet models
const RESNET_SHORTCUTS = Dict(:A => (downsample_identity, downsample_identity),
                              :B => (downsample_conv, downsample_identity),
                              :C => (downsample_conv, downsample_conv),
                              :D => (downsample_pool, downsample_identity))

# Stride for each block in the ResNet model
function resnet_stride(stage_idx::Integer, block_idx::Integer)
    return stage_idx == 1 || block_idx != 1 ? 1 : 2
end

# returns `DropBlock`s for each stage of the ResNet as in timm.
# TODO - add experimental options for DropBlock as part of the API (#188)
# function _drop_blocks(drop_block_rate::AbstractFloat)
#     return [
#         identity, identity,
#         DropBlock(drop_block_rate, 5, 0.25), DropBlock(drop_block_rate, 3, 1.00),
#     ]
# end

"""
    resnet_stem(; stem_type = :default, inchannels::Integer = 3, replace_stem_pool = false,
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

  - `inchannels`: number of input channels
  - `replace_pool`: Set to true to replace the max pooling layers with a 3x3 convolution +
    normalization with a stride of two.
  - `norm_layer`: The normalisation layer used in the stem.
  - `activation`: The activation function used in the stem.
"""
function resnet_stem(stem_type::Symbol = :default; inchannels::Integer = 3,
                     replace_pool::Bool = false, activation = relu,
                     norm_layer = BatchNorm, revnorm::Bool = false)
    _checkconfig(stem_type, [:default, :deep, :deep_tiered])
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
                                norm_layer, revnorm, stride = 2, pad = 1)...,
                      conv_norm((3, 3), stem_channels[1] => stem_channels[2], activation;
                                norm_layer, pad = 1)...,
                      Conv((3, 3), stem_channels[2] => inplanes; pad = 1, bias = false))
    else
        conv1 = Conv((7, 7), inchannels => inplanes; stride = 2, pad = 3, bias = false)
    end
    bn1 = norm_layer(inplanes, activation)
    # Stem pooling
    stempool = replace_pool ?
               Chain(conv_norm((3, 3), inplanes => inplanes, activation; norm_layer,
                               revnorm, stride = 2, pad = 1)...) :
               MaxPool((3, 3); stride = 2, pad = 1)
    return Chain(conv1, bn1, stempool)
end

function resnet_planes(block_repeats::AbstractVector{<:Integer})
    return Iterators.flatten((64 * 2^(stage_idx - 1) for _ in 1:stages)
                             for (stage_idx, stages) in enumerate(block_repeats))
end

function resnet(img_dims, stem, get_layers, block_repeats::AbstractVector{<:Integer},
                connection, classifier_fn)
    # Build stages of the ResNet
    stage_blocks = cnn_stages(get_layers, block_repeats, connection)
    backbone = Chain(stem, stage_blocks...)
    # Add classifier to the backbone
    nfeaturemaps = Flux.outputsize(backbone, img_dims; padbatch = true)[3]
    return Chain(backbone, classifier_fn(nfeaturemaps))
end

function resnet(block_type, block_repeats::AbstractVector{<:Integer},
                downsample_opt::NTuple{2, Any} = (downsample_conv, downsample_identity);
                cardinality::Integer = 1, base_width::Integer = 64,
                inplanes::Integer = 64,
                reduction_factor::Integer = 1, imsize::Dims{2} = (256, 256),
                inchannels::Integer = 3, stem_fn = resnet_stem, connection = addact,
                activation = relu, norm_layer = BatchNorm, revnorm::Bool = false,
                attn_fn = planes -> identity, pool_layer = AdaptiveMeanPool((1, 1)),
                use_conv::Bool = false, drop_block_rate = nothing, drop_path_rate = nothing,
                dropout_rate = nothing, nclasses::Integer = 1000, kwargs...)
    # Build stem
    stem = stem_fn(; inchannels)
    # Block builder
    if block_type == basicblock
        @assert cardinality==1 "Cardinality must be 1 for `basicblock`"
        @assert base_width==64 "Base width must be 64 for `basicblock`"
        get_layers = basicblock_builder(block_repeats; inplanes, reduction_factor,
                                        activation, norm_layer, revnorm, attn_fn,
                                        drop_block_rate, drop_path_rate,
                                        stride_fn = resnet_stride,
                                        planes_fn = resnet_planes,
                                        downsample_tuple = downsample_opt, kwargs...)
    elseif block_type == bottleneck
        get_layers = bottleneck_builder(block_repeats; inplanes, cardinality, base_width,
                                        reduction_factor, activation, norm_layer, revnorm,
                                        attn_fn, drop_block_rate, drop_path_rate,
                                        stride_fn = resnet_stride,
                                        planes_fn = resnet_planes,
                                        downsample_tuple = downsample_opt, kwargs...)
    elseif block_type == bottle2neck
        @assert isnothing(drop_block_rate) "DropBlock not supported for `bottle2neck`.
        Set `drop_block_rate` to nothing."
        @assert isnothing(drop_path_rate) "DropPath not supported for `bottle2neck`.
        Set `drop_path_rate` to nothing."
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
    classifier_fn = nfeatures -> create_classifier(nfeatures, nclasses; dropout_rate,
                                                   pool_layer, use_conv)
    return resnet((imsize..., inchannels), stem, get_layers, block_repeats,
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
