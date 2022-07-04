"""
    AdaptiveMeanMaxPool(output_size = (1, 1); connection = .+)

A type of adaptive pooling layer which uses both mean and max pooling and combines them to
produce a single output. Note that this is equivalent to
`Parallel(connection, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))`

# Arguments

  - `output_size`: The size of the output after pooling.
  - `connection`: The connection type to use.
"""
function AdaptiveMeanMaxPool(output_size = (1, 1); connection = .+)
    return Parallel(connection, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))
end

"""
    SelectAdaptivePool(output_size = (1, 1); pool_type = :mean, flatten = false)

Adaptive pooling factory function.

# Arguments

  - `output_size`: The size of the output after pooling.
  - `pool_type`: The type of adaptive pooling to use. One of `:mean`, `:max`, `:meanmax`,
    `:catmeanmax` or `:identity`.
  - `flatten`: Whether to flatten the output from the pooling layer.
"""
function SelectAdaptivePool(output_size = (1, 1); pool_type = :mean, flatten = false)
    if pool_type == :mean
        pool = AdaptiveMeanPool(output_size)
    elseif pool_type == :max
        pool = AdaptiveMaxPool(output_size)
    elseif pool_type == :meanmax
        pool = 0.5f0 * AdaptiveMeanMaxPool(output_size)
    elseif pool_type == :catmeanmax
        pool = AdaptiveMeanMaxPool(output_size; connection = cat_channels)
    elseif pool_type == :identity
        pool = identity
    else
        throw(AssertionError("Invalid pool type: $pool_type"))
    end
    flatten_fn = flatten ? MLUtils.flatten : identity
    return Chain(pool, flatten_fn)
end
