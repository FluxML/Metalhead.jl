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

# A failure to reach the Hub, rather than an HTTP error status (code 0).
is_unreachable(e) = e isa RequestError && e.code != 0

# Offline (`HF_HUB_OFFLINE`) or when the Hub can't be reached, HuggingFaceApi can only tell
# that a file isn't in the cache, not whether the repository has it.
function is_uncached(e, repo, file, revision)
    HF.offline_mode() || is_unreachable(e) || return false
    return HF.try_to_load_from_cache(HF.HuggingFaceURL(repo, file; revision)) === nothing
end

# The first of `files` in `repo`, as `file => local path`. Offline or without network, a
# file missing from the cache is skipped like one the repository lacks. If none is found,
# the error says which files were tried (`what` names them) and why.
function hf_download_first(repo, files, what; revision = "main")
    unreachable = nothing
    for file in files
        try
            return file => hf_download(repo, file; revision)
        catch e
            is_not_found(e) || is_uncached(e, repo, file, revision) || rethrow()
            is_unreachable(e) && (unreachable = e)
        end
    end
    tried = "tried $(join(files, ", ", " and "))"
    uncached = "$repo@$revision has no $what in the cache ($tried)"
    HF.offline_mode() && error(uncached, ", and HF_HUB_OFFLINE is set")
    unreachable === nothing ||
        error(uncached, ", and the Hub can't be reached: ", unreachable.message)
    return error("$repo@$revision has no $what ($tried)")
end

"""
    hf_weights(repo; revision = "main", subfolder = "", name = "model") -> Vector{String}

Download the safetensors weights `name.safetensors` of `repo` (a single file or all shards of
`name.safetensors.index.json`), in `subfolder`, returning their local paths. diffusers names
its weights `diffusion_pytorch_model`.
"""
function hf_weights(repo::AbstractString; revision::AbstractString = "main",
        subfolder::AbstractString = "", name::AbstractString = "model")
    path(file) = join_path(subfolder, file)
    file, local_path = hf_download_first(repo,
        [path("$name.safetensors"), path("$name.safetensors.index.json")],
        "safetensors weights"; revision)
    endswith(file, ".json") || return [local_path]
    index = JSON.parsefile(local_path)
    return [hf_download(repo, path(s); revision) for s in shard_files(index)]
end

shard_files(index) = sort!(unique!(string.(values(index["weight_map"]))))
