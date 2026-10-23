"""
    AdaptiveMeanMaxPool([connection = +], output_size::Tuple = (1, 1))

A type of adaptive pooling layer which uses both mean and max pooling and combines them to
produce a single output. Note that this is equivalent to
`Parallel(connection, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))`.
When `connection` is not specified, it defaults to `+`.

# Arguments

  - `connection`: The connection type to use.
  - `output_size`: The size of the output after pooling.
"""
function AdaptiveMeanMaxPool(connection, output_size::Tuple = (1, 1))
    return Parallel(connection, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))
end
AdaptiveMeanMaxPool(output_size::Tuple = (1, 1)) = AdaptiveMeanMaxPool(+, output_size)
