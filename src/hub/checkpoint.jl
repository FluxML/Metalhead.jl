"""
    Checkpoint(paths)

Lazy view over one or more safetensors files (e.g. the shards of a sharded checkpoint).
Indexing with a PyTorch `state_dict` key returns the tensor in Julia layout,
i.e. with the PyTorch dimensions reversed (NCHW → WHCN).
"""
struct Checkpoint
    files::Vector{SafeTensors.SafeTensor}
    index::Dict{String, Int}
end

function Checkpoint(paths::AbstractVector{<:AbstractString})
    files = [SafeTensors.deserialize(p) for p in paths]
    index = Dict{String, Int}()
    for (i, f) in enumerate(files), k in keys(f)
        haskey(index, k) && error("tensor $k appears in more than one checkpoint file")
        index[k] = i
    end
    return Checkpoint(files, index)
end
Checkpoint(path::AbstractString) = Checkpoint([path])

Base.keys(c::Checkpoint) = keys(c.index)
Base.haskey(c::Checkpoint, k) = haskey(c.index, k)
Base.getindex(c::Checkpoint, k::AbstractString) = torch2julia(c.files[c.index[k]][k])
torch_shape(c::Checkpoint, k::AbstractString) = size(c.files[c.index[k]][k])

torch2julia(x::AbstractArray) = permutedims(x, ndims(x):-1:1)
