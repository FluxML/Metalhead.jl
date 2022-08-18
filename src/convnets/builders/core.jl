# TODO potentially refactor other CNNs to use this
function cnn_stages(get_layers, block_repeats::AbstractVector{<:Integer}, connection)
    # Construct each stage
    stages = []
    for (stage_idx, nblocks) in enumerate(block_repeats)
        # Construct the blocks for each stage
        blocks = map(1:nblocks) do block_idx
            branches = get_layers(stage_idx, block_idx)
            return length(branches) == 1 ? only(branches) :
                   Parallel(connection, branches...)
        end
        push!(stages, Chain(blocks...))
    end
    return stages
end

function cnn_stages(get_layers, block_repeats::AbstractVector{<:Integer})
    # Construct each stage
    stages = []
    for (stage_idx, nblocks) in enumerate(block_repeats)
        # Construct the blocks for each stage
        blocks = map(1:nblocks) do block_idx
            branches = get_layers(stage_idx, block_idx)
            @assert length(branches)==1 "get_layers should return a single branch for each
          block if no connection is specified"
            return only(branches)
        end
        push!(stages, Chain(blocks...))
    end
    return stages
end
