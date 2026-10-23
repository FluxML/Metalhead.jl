# Hub access through HuggingFaceHub.jl. Files are cached per commit sha,
# so a pinned revision needs no network once cached.

hub_repo(repo, revision) = HF.Model(; id = String(repo), revision = String(revision))

"""
    hf_download(repo, filename; revision = "main") -> String

Download `filename` from the Hub model repository `repo` at `revision`, returning the local path.
"""
function hf_download(repo::AbstractString, filename::AbstractString; revision::AbstractString = "main")
    return HF.file_download(hub_repo(repo, revision), filename)
end

is_not_found(e) = hasproperty(e, :status) && getproperty(e, :status) == 404

"""
    hf_weights(repo; revision = "main") -> Vector{String}

Download the safetensors weights of `repo` (a single file or all shards), returning their local paths.
"""
function hf_weights(repo::AbstractString; revision::AbstractString = "main")
    try
        return [hf_download(repo, "model.safetensors"; revision)]
    catch e
        is_not_found(e) || rethrow()
    end
    index = try
        JSON.parsefile(hf_download(repo, "model.safetensors.index.json"; revision))
    catch e
        is_not_found(e) || rethrow()
        error("$repo@$revision has no safetensors weights (model.safetensors or model.safetensors.index.json)")
    end
    shards = sort!(unique(string.(values(index["weight_map"]))))
    return [hf_download(repo, s; revision) for s in shards]
end
