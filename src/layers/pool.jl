function AdaptiveMeanMaxPool(output_size = (1, 1))
    return 0.5 * Parallel(.+, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))
end

function AdaptiveCatMeanMaxPool(output_size = (1, 1))
    return Parallel(cat_channels, AdaptiveAvgMaxPool(output_size),
                    AdaptiveMaxPool(output_size))
end

function SelectAdaptivePool(output_size = (1, 1); pool_type = :mean, flatten = false)
    if pool_type == :mean
        pool = AdaptiveAvgPool(output_size)
    elseif pool_type == :max
        pool = AdaptiveMaxPool(output_size)
    elseif pool_type == :meanmax
        pool = AdaptiveAvgMaxPool(output_size)
    elseif pool_type == :catmeanmax
        pool = AdaptiveCatAvgMaxPool(output_size)
    elseif pool_type = :identity
        pool = identity
    else
        throw(AssertionError("Invalid pool type: $pool_type"))
    end
    flatten_fn = flatten ? MLUtils.flatten : identity
    return Chain(pool, flatten_fn)
end
