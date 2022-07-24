"""
    AdaptiveMeanMaxPool(output_size = (1, 1); connection = +)

A type of adaptive pooling layer which uses both mean and max pooling and combines them to
produce a single output. Note that this is equivalent to
`Parallel(connection, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))`

# Arguments

  - `output_size`: The size of the output after pooling.
  - `connection`: The connection type to use.
"""
function AdaptiveMeanMaxPool(output_size = (1, 1); connection = +)
    return Parallel(connection, AdaptiveMeanPool(output_size), AdaptiveMaxPool(output_size))
end
