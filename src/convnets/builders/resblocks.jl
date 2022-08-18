function basicblock_builder(block_repeats::AbstractVector{<:Integer};
                            inplanes::Integer = 64, reduction_factor::Integer = 1,
                            expansion::Integer = 1, norm_layer = BatchNorm,
                            revnorm::Bool = false, activation = relu,
                            attn_fn = planes -> identity,
                            drop_block_rate = nothing, drop_path_rate = nothing,
                            stride_fn = resnet_stride, planes_fn = resnet_planes,
                            downsample_tuple = (downsample_conv, downsample_identity))
    # DropBlock, DropPath both take in rates based on a linear scaling schedule
    # Also get `planes_vec` needed for block `inplanes` and `planes` calculations
    pathschedule = linear_scheduler(drop_path_rate; depth = sum(block_repeats))
    blockschedule = linear_scheduler(drop_block_rate; depth = sum(block_repeats))
    planes_vec = collect(planes_fn(block_repeats))
    # closure over `idxs`
    function get_layers(stage_idx::Integer, block_idx::Integer)
        # DropBlock, DropPath both take in rates based on a linear scaling schedule
        # This is also needed for block `inplanes` and `planes` calculations
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        planes = planes_vec[schedule_idx]
        inplanes = schedule_idx == 1 ? inplanes : planes_vec[schedule_idx - 1] * expansion
        # `resnet_stride` is a callback that the user can tweak to change the stride of the
        # blocks. It defaults to the standard behaviour as in the paper
        stride = stride_fn(stage_idx, block_idx)
        downsample_fn = stride != 1 || inplanes != planes * expansion ?
                        downsample_tuple[1] : downsample_tuple[2]
        drop_path = DropPath(pathschedule[schedule_idx])
        drop_block = DropBlock(blockschedule[schedule_idx])
        block = basicblock(inplanes, planes; stride, reduction_factor, activation,
                           norm_layer, revnorm, attn_fn, drop_path, drop_block)
        downsample = downsample_fn(inplanes, planes * expansion; stride, norm_layer,
                                   revnorm)
        return block, downsample
    end
    return get_layers
end

function bottleneck_builder(block_repeats::AbstractVector{<:Integer};
                            inplanes::Integer = 64, cardinality::Integer = 1,
                            base_width::Integer = 64, reduction_factor::Integer = 1,
                            expansion::Integer = 4, norm_layer = BatchNorm,
                            revnorm::Bool = false, activation = relu,
                            attn_fn = planes -> identity,
                            drop_block_rate = nothing, drop_path_rate = nothing,
                            stride_fn = resnet_stride, planes_fn = resnet_planes,
                            downsample_tuple = (downsample_conv, downsample_identity))
    pathschedule = linear_scheduler(drop_path_rate; depth = sum(block_repeats))
    blockschedule = linear_scheduler(drop_block_rate; depth = sum(block_repeats))
    planes_vec = collect(planes_fn(block_repeats))
    # closure over `idxs`
    function get_layers(stage_idx::Integer, block_idx::Integer)
        # DropBlock, DropPath both take in rates based on a linear scaling schedule
        # This is also needed for block `inplanes` and `planes` calculations
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        planes = planes_vec[schedule_idx]
        inplanes = schedule_idx == 1 ? inplanes : planes_vec[schedule_idx - 1] * expansion
        # `resnet_stride` is a callback that the user can tweak to change the stride of the
        # blocks. It defaults to the standard behaviour as in the paper
        stride = stride_fn(stage_idx, block_idx)
        downsample_fn = stride != 1 || inplanes != planes * expansion ?
                        downsample_tuple[1] : downsample_tuple[2]
        drop_path = DropPath(pathschedule[schedule_idx])
        drop_block = DropBlock(blockschedule[schedule_idx])
        block = bottleneck(inplanes, planes; stride, cardinality, base_width,
                           reduction_factor, activation, norm_layer, revnorm,
                           attn_fn, drop_path, drop_block)
        downsample = downsample_fn(inplanes, planes * expansion; stride, norm_layer,
                                   revnorm)
        return block, downsample
    end
    return get_layers
end
