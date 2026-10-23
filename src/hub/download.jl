# Hub access through HuggingFaceApi.jl, which shares the download cache of Python's
# huggingface_hub. Files are cached per commit sha, so a pinned revision needs no network once cached.

"""
    hf_download(repo, filename; revision = "main") -> String

Download `filename` from the Hub model repository `repo` at `revision`, returning the local path.
"""
function hf_download(repo::AbstractString, filename::AbstractString; revision::AbstractString = "main")
    return HF.hf_hub_download(repo, filename; revision)
end

# HuggingFaceApi reports a missing file as an ErrorException with the Hub's error code.
is_not_found(e) = e isa ErrorException && occursin("EntryNotFound", e.msg)

"""
    hf_weights(repo; revision = "main", subfolder = "", name = "model") -> Vector{String}

Download the safetensors weights `name.safetensors` of `repo` (a single file or all shards of
`name.safetensors.index.json`), in `subfolder`, returning their local paths. diffusers names
its weights `diffusion_pytorch_model`.
"""
function hf_weights(repo::AbstractString; revision::AbstractString = "main",
        subfolder::AbstractString = "", name::AbstractString = "model")
    path(file) = join_path(subfolder, file)
    try
        return [hf_download(repo, path("$name.safetensors"); revision)]
    catch e
        is_not_found(e) || rethrow()
    end
    index = try
        JSON.parsefile(hf_download(repo, path("$name.safetensors.index.json"); revision))
    catch e
        is_not_found(e) || rethrow()
        error("$repo@$revision has no safetensors weights ($(path(name)).safetensors or $(path(name)).safetensors.index.json)")
    end
    return [hf_download(repo, path(s); revision) for s in shard_files(index)]
end

shard_files(index) = sort!(unique!(string.(values(index["weight_map"]))))
