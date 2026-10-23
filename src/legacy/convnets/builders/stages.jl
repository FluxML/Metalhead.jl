"""
    cnn_stages(get_layers, block_repeats::AbstractVector{<:Integer}, [connection = nothing])

Creates a convolutional neural network backbone by calling the function `get_layers`
repeatedly.

# Arguments

  - `get_layers` is a function that takes in two inputs - the `stage_idx`, or the index of
    the stage, and the `block_idx`, or the index of the block within the stage. It returns a
    tuple of layers. If the tuple returned by `get_layers` has more than one element, then
    `connection` is used - if not, then the only element of the tuple is directly inserted
    into the network.
  - `block_repeats` is a `Vector` of integers, where each element specifies the number of
    times the `get_layers` function should be called for that stage.
  - `connection` defaults to `nothing` and is an optional argument that specifies the
    connection type between the layers. It is passed to `Parallel` and is useful for residual
    network structures. For example, for ResNet, the connection is simply `+`. If `connection`
    is `nothing`, then every call to `get_layers` _must_ return a tuple of length 1.
"""
function cnn_stages(get_layers, block_repeats::AbstractVector{<:Integer},
                    connection = nothing)
    # Construct each stage
    stages = []
    for (stage_idx, nblocks) in enumerate(block_repeats)
        # Construct the blocks for each stage
        blocks = map(1:nblocks) do block_idx
            branches = get_layers(stage_idx, block_idx)
            if isnothing(connection)
                @assert length(branches)==1 "get_layers should return a single branch for
                each block if no connection is specified"
            end
            return length(branches) == 1 ? only(branches) :
                   Parallel(connection, branches...)
        end
        push!(stages, Chain(blocks...))
    end
    return stages
end
