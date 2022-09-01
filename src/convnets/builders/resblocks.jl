"""
    basicblock_builder(block_repeats::AbstractVector{<:Integer};
                       inplanes::Integer = 64, reduction_factor::Integer = 1,
                       expansion::Integer = 1, norm_layer = BatchNorm,
                       revnorm::Bool = false, activation = relu,
                       attn_fn = planes -> identity,
                       dropblock_prob = nothing, stochastic_depth_prob = nothing,
                       stride_fn = resnet_stride, planes_fn = resnet_planes,
                       downsample_tuple = (downsample_conv, downsample_identity))

Builder for creating a basic block for a ResNet model.
([reference](https://arxiv.org/abs/1512.03385))

# Arguments

  - `block_repeats`: number of repeats of a block in each stage

  - `inplanes`: number of input channels
  - `reduction_factor`: reduction factor for the number of channels in each stage
  - `expansion`: expansion factor for the number of channels for the block
  - `norm_layer`: normalization layer to use
  - `revnorm`: set to `true` to place normalization layer before the convolution
  - `activation`: activation function to use
  - `attn_fn`: attention function to use
  - `dropblock_prob`: dropblock probability. Set to `nothing` to disable `DropBlock`
  - `stochastic_depth_prob`: stochastic depth probability. Set to `nothing` to disable `StochasticDepth`
  - `stride_fn`: callback for computing the stride of the block
  - `planes_fn`: callback for computing the number of channels in each block
  - `downsample_tuple`: two-element tuple of downsample functions to use. The first one
    is used when the number of channels changes in the block, the second one is used
    when the number of channels stays the same.
"""
function basicblock_builder(block_repeats::AbstractVector{<:Integer};
                            inplanes::Integer = 64, reduction_factor::Integer = 1,
                            expansion::Integer = 1, norm_layer = BatchNorm,
                            revnorm::Bool = false, activation = relu,
                            attn_fn = planes -> identity, dropblock_prob = nothing,
                            stochastic_depth_prob = nothing, stride_fn = resnet_stride,
                            planes_fn = resnet_planes,
                            downsample_tuple = (downsample_conv, downsample_identity))
    # DropBlock, StochasticDepth both take in probabilities based on a linear scaling schedule
    # Also get `planes_vec` needed for block `inplanes` and `planes` calculations
    sdschedule = linear_scheduler(stochastic_depth_prob; depth = sum(block_repeats))
    dbschedule = linear_scheduler(dropblock_prob; depth = sum(block_repeats))
    planes_vec = collect(planes_fn(block_repeats))
    # closure over `idxs`
    function get_layers(stage_idx::Integer, block_idx::Integer)
        # DropBlock, StochasticDepth both take in probabilities based on a linear scaling schedule
        # This is also needed for block `inplanes` and `planes` calculations
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        planes = planes_vec[schedule_idx]
        inplanes = schedule_idx == 1 ? inplanes : planes_vec[schedule_idx - 1] * expansion
        stride = stride_fn(stage_idx, block_idx)
        downsample_fn = stride != 1 || inplanes != planes * expansion ?
                        downsample_tuple[1] : downsample_tuple[2]
        drop_path = StochasticDepth(sdschedule[schedule_idx])
        drop_block = DropBlock(dbschedule[schedule_idx])
        block = basicblock(inplanes, planes; stride, reduction_factor, activation,
                           norm_layer, revnorm, attn_fn, drop_path, drop_block)
        downsample = downsample_fn(inplanes, planes * expansion; stride, norm_layer,
                                   revnorm)
        return (downsample, block)
    end
    return get_layers
end

"""
    bottleneck_builder(block_repeats::AbstractVector{<:Integer};
                       inplanes::Integer = 64, cardinality::Integer = 1,
                       base_width::Integer = 64, reduction_factor::Integer = 1,
                       expansion::Integer = 4, norm_layer = BatchNorm,
                       revnorm::Bool = false, activation = relu,
                       attn_fn = planes -> identity, dropblock_prob = nothing,
                       stochastic_depth_prob = nothing, stride_fn = resnet_stride,
                       planes_fn = resnet_planes,
                       downsample_tuple = (downsample_conv, downsample_identity))

Builder for creating a bottleneck block for a ResNet/ResNeXt model.
([reference](https://arxiv.org/abs/1611.05431))

# Arguments

  - `block_repeats`: number of repeats of a block in each stage
  - `inplanes`: number of input channels
  - `cardinality`: number of groups for the convolutional layer
  - `base_width`: base width for the convolutional layer
  - `reduction_factor`: reduction factor for the number of channels in each stage
  - `expansion`: expansion factor for the number of channels for the block
  - `norm_layer`: normalization layer to use
  - `revnorm`: set to `true` to place normalization layer before the convolution
  - `activation`: activation function to use
  - `attn_fn`: attention function to use
  - `dropblock_prob`: dropblock probability. Set to `nothing` to disable `DropBlock`
  - `stochastic_depth_prob`: stochastic depth probability. Set to `nothing` to disable `StochasticDepth`
  - `stride_fn`: callback for computing the stride of the block
  - `planes_fn`: callback for computing the number of channels in each block
  - `downsample_tuple`: two-element tuple of downsample functions to use. The first one
    is used when the number of channels changes in the block, the second one is used
    when the number of channels stays the same.
"""
function bottleneck_builder(block_repeats::AbstractVector{<:Integer};
                            inplanes::Integer = 64, cardinality::Integer = 1,
                            base_width::Integer = 64, reduction_factor::Integer = 1,
                            expansion::Integer = 4, norm_layer = BatchNorm,
                            revnorm::Bool = false, activation = relu,
                            attn_fn = planes -> identity, dropblock_prob = nothing,
                            stochastic_depth_prob = nothing, stride_fn = resnet_stride,
                            planes_fn = resnet_planes,
                            downsample_tuple = (downsample_conv, downsample_identity))
    sdschedule = linear_scheduler(stochastic_depth_prob; depth = sum(block_repeats))
    dbschedule = linear_scheduler(dropblock_prob; depth = sum(block_repeats))
    planes_vec = collect(planes_fn(block_repeats))
    # closure over `idxs`
    function get_layers(stage_idx::Integer, block_idx::Integer)
        # DropBlock, StochasticDepth both take in rates based on a linear scaling schedule
        # This is also needed for block `inplanes` and `planes` calculations
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        planes = planes_vec[schedule_idx]
        inplanes = schedule_idx == 1 ? inplanes : planes_vec[schedule_idx - 1] * expansion
        stride = stride_fn(stage_idx, block_idx)
        downsample_fn = stride != 1 || inplanes != planes * expansion ?
                        downsample_tuple[1] : downsample_tuple[2]
        drop_path = StochasticDepth(sdschedule[schedule_idx])
        drop_block = DropBlock(dbschedule[schedule_idx])
        block = bottleneck(inplanes, planes; stride, cardinality, base_width,
                           reduction_factor, activation, norm_layer, revnorm,
                           attn_fn, drop_path, drop_block)
        downsample = downsample_fn(inplanes, planes * expansion; stride, norm_layer,
                                   revnorm)
        return (downsample, block)
    end
    return get_layers
end

"""
    bottle2neck_builder(block_repeats::AbstractVector{<:Integer};
                        inplanes::Integer = 64, cardinality::Integer = 1,
                        base_width::Integer = 26, scale::Integer = 4,
                        expansion::Integer = 4, norm_layer = BatchNorm,
                        revnorm::Bool = false, activation = relu,
                        attn_fn = planes -> identity, stride_fn = resnet_stride,
                        planes_fn = resnet_planes,
                        downsample_tuple = (downsample_conv, downsample_identity))

Builder for creating a bottle2neck block for a Res2Net model.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `block_repeats`: number of repeats of a block in each stage
  - `inplanes`: number of input channels
  - `cardinality`: number of groups for the convolutional layer
  - `base_width`: base width for the convolutional layer
  - `scale`: scale for the number of channels in each block
  - `expansion`: expansion factor for the number of channels for the block
  - `norm_layer`: normalization layer to use
  - `revnorm`: set to `true` to place normalization layer before the convolution
  - `activation`: activation function to use
  - `attn_fn`: attention function to use
  - `stride_fn`: callback for computing the stride of the block
  - `planes_fn`: callback for computing the number of channels in each block
  - `downsample_tuple`: two-element tuple of downsample functions to use. The first one
    is used when the number of channels changes in the block, the second one is used
    when the number of channels stays the same.
"""
function bottle2neck_builder(block_repeats::AbstractVector{<:Integer};
                             inplanes::Integer = 64, cardinality::Integer = 1,
                             base_width::Integer = 26, scale::Integer = 4,
                             expansion::Integer = 4, norm_layer = BatchNorm,
                             revnorm::Bool = false, activation = relu,
                             attn_fn = planes -> identity, stride_fn = resnet_stride,
                             planes_fn = resnet_planes,
                             downsample_tuple = (downsample_conv, downsample_identity))
    planes_vec = collect(planes_fn(block_repeats))
    # closure over `idxs`
    function get_layers(stage_idx::Integer, block_idx::Integer)
        # This is needed for block `inplanes` and `planes` calculations
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        planes = planes_vec[schedule_idx]
        inplanes = schedule_idx == 1 ? inplanes : planes_vec[schedule_idx - 1] * expansion
        stride = stride_fn(stage_idx, block_idx)
        downsample_fn = (stride != 1 || inplanes != planes * expansion) ?
                        downsample_tuple[1] : downsample_tuple[2]
        is_first = (stride > 1 || downsample_fn != downsample_tuple[2]) ? true : false
        block = bottle2neck(inplanes, planes; stride, cardinality, base_width, scale,
                            activation, is_first, norm_layer, revnorm, attn_fn)
        downsample = downsample_fn(inplanes, planes * expansion; stride, norm_layer,
                                   revnorm)
        return block, downsample
    end
    return get_layers
end
