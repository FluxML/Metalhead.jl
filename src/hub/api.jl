const ELTYPE_CONVERTERS = Dict{DataType, Function}(
    Float16 => f16, BFloat16 => bf16, Float32 => f32, Float64 => f64)

nt(d::AbstractDict) = (; (Symbol(k) => jsonvalue(v) for (k, v) in d)...)
jsonvalue(v::AbstractVector) = Tuple(jsonvalue.(v))
jsonvalue(v::AbstractDict) = nt(v)
jsonvalue(v) = v

# A model source: a Hub repository, or a local directory laid out like one, and the subfolder
# of a component of a diffusers pipeline ("" for the root).
struct ModelSource
    repo::Union{Nothing, String}
    dir::Union{Nothing, String}
    revision::String
    subfolder::String
end

function subsource(s::ModelSource, name)
    return ModelSource(s.repo, s.dir, s.revision, join_path(s.subfolder, name))
end
join_path(a, b) = isempty(a) ? String(b) : isempty(b) ? String(a) : string(a, "/", b)

function source_file(s::ModelSource, file)
    file = join_path(s.subfolder, file)
    return s.dir === nothing ? hf_download(s.repo, file; revision = s.revision) :
           joinpath(s.dir, file)
end

function source_weights(s::ModelSource, name = "model")
    s.dir === nothing &&
        return hf_weights(s.repo; revision = s.revision, subfolder = s.subfolder, name)
    dir = joinpath(s.dir, s.subfolder)
    file = joinpath(dir, "$name.safetensors")
    isfile(file) && return [file]
    index = joinpath(dir, "$name.safetensors.index.json")
    isfile(index) || error("$dir has no safetensors weights ($name.safetensors or $name.safetensors.index.json)")
    return [joinpath(dir, f) for f in shard_files(JSON.parsefile(index))]
end

# The files that say what model a source holds, by precedence: the `modules.json` of a
# sentence-transformers model, which also has the config.json of its transformers model; the
# `config.json` of a model; the `model_index.json` of a diffusers pipeline.
const CONFIG_FILES = ("modules.json", "config.json", "model_index.json")

# A parsed configuration file `name`, as `find_format` takes it: a modules.json, a list of
# modules, as `Dict("modules.json" => modules)`.
function config_of(name, parsed)
    return name == "modules.json" ? Dict{String, Any}(name => parsed) : parsed
end

# The configuration of a source, from the first of `CONFIG_FILES` that it has.
function source_config(s::ModelSource)
    files = [join_path(s.subfolder, f) for f in CONFIG_FILES]
    if s.dir === nothing
        file, path = hf_download_first(s.repo, files, "configuration";
            revision = s.revision)
        return config_of(basename(file), JSON.parsefile(path))
    end
    for (name, file) in zip(CONFIG_FILES, files)
        path = joinpath(s.dir, file)
        isfile(path) && return config_of(name, JSON.parsefile(path))
    end
    return error("$(join_path(s.dir, s.subfolder)) has no modules.json, config.json or model_index.json")
end

function bare_spec(name)
    haskey(REGISTRY, name) && return REGISTRY[name]
    return error("`$name` is neither a ported architecture (see `list_models()`), a Hub repository (`org/name`) nor a directory")
end

# A Hub repository `org/name`, and the subfolder of `org/name/subfolder`.
function split_repo(name)
    parts = split(name, '/')
    return join(parts[1:2], '/'), join(parts[3:end], '/')
end

# What `name` stands for: a local directory, a Hub repository (or a subfolder of one), or the
# repository of the default checkpoint of a bare architecture. A validated repository defaults
# to the commit it was validated at.
function model_source(name::AbstractString, revision, subfolder = "")
    isdir(name) && return ModelSource(nothing, abspath(name), "", subfolder)
    if !occursin('/', name)
        spec = bare_spec(name)
        isempty(spec.references) &&
            error("`$name` has no validated pretrained checkpoint; pass a Hub repository instead")
        name, sha = first(spec.references)
        revision = something(revision, sha)
    end
    repo, sub = split_repo(name)
    pinned = [sha for spec in values(REGISTRY) for (ref, sha) in spec.references
              if first(split_repo(ref)) == repo]
    revision = something(revision, isempty(pinned) ? "main" : first(pinned))
    return ModelSource(String(repo), nothing, revision, join_path(sub, subfolder))
end

"""
    create_model(name; pretrained, revision = nothing, subfolder = "", eltype = Float32, kwargs...)

Build a ported model.

`name` is a Hub repository (`"timm/resnet50.a1_in1k"`, `"HuggingFaceTB/SmolLM2-135M"`), a bare
architecture (`"resnet50"`, `"LlamaForCausalLM"`; see [`list_models`](@ref)), or a local
directory containing a `config.json` and `model.safetensors`.

A repository or a directory names a checkpoint, whose weights are loaded: `pretrained`
defaults to `true` for them. A bare architecture names the architecture alone, whose
weights are random and for which nothing is downloaded: `pretrained` defaults to `false`,
and `pretrained = true` loads the weights of its default validated checkpoint.
`pretrained = false` builds the architecture of a repository or directory with random
weights, downloading only its `config.json`. `revision` selects a branch, tag or commit of
the repository; validated checkpoints default to the commit they were validated at.
The remaining keyword arguments are the original library's model arguments, e.g. timm's
`num_classes`, or the fields of a transformers model's `config.json`, e.g. `num_hidden_layers`.
Changing `num_classes` of a pretrained model re-initialises its classifier.

A diffusers repository holds a pipeline, described by its `model_index.json`, whose
components are in subfolders: `create_model` builds the whole pipeline, and with `subfolder`
(e.g. `"vae"` or `"transformer"`) a single component, which `"org/name/vae"` also names.

A sentence-transformers repository, with a `modules.json`, holds an embedding model, the
modules of a pipeline over a transformers model: `create_model` builds a
[`SentenceTransformer`](@ref Metalhead.SentenceTransformers.SentenceTransformer), whatever
the `config.json` of its transformers model names, and its keyword arguments are the
settings of the model, such as `truncate_dim`.

The model is returned in test mode; call `Flux.trainmode!` before training.
"""
function create_model(name::AbstractString; pretrained::Union{Nothing, Bool} = nothing,
        revision::Union{Nothing, AbstractString} = nothing, subfolder::AbstractString = "",
        eltype::Type{<:AbstractFloat} = Float32, kwargs...)
    bare = !isdir(name) && !occursin('/', name)
    pretrained = something(pretrained, !bare)
    if !pretrained && bare
        isempty(subfolder) || error("`subfolder` needs a Hub repository or a directory")
        spec = bare_spec(name)
        model = spec.build(; merge(spec.defaults, values(kwargs))...)
        return Flux.testmode!(ELTYPE_CONVERTERS[eltype](model))
    end
    source = model_source(name, revision, String(subfolder))
    config = source_config(source)
    format, architecture = identify(config)
    return instantiate(format, lookup(architecture), source, config; pretrained, eltype,
        kwargs...)
end

"""
    forward_features(model, x)

Run `model` up to its last feature map, before pooling and classification (timm's `forward_features`).

This and [`forward_head`](@ref), [`forward_intermediates`](@ref) and [`feature_info`](@ref) are
the interface of image backbones and classifiers, and of language models.
"""
function forward_features end

"""
    forward_head(model, features; pre_logits = false)

Apply the pooling and classifier of `model` to the output of [`forward_features`](@ref).
With `pre_logits = true`, return the pooled features before the classifier.
"""
function forward_head end

"""
    forward_intermediates(model, x; indices = nothing, norm = false, stop_early = false,
                          output_fmt = "NCHW", intermediates_only = false)

Run `model` and collect the outputs of its stages or blocks, for dense prediction heads
(timm's `forward_intermediates`). The intermediates are those of [`feature_info`](@ref).

  - `indices`: `nothing` takes all intermediates, an integer `k` the last `k`, and a collection
    the given positions (1-based, unlike timm's).
  - `norm`: apply the final norm of the model to the intermediates, where timm does.
  - `stop_early`: skip the stages after the last selected one.
  - `output_fmt`: `"NCHW"` returns WHCN feature maps (the Julia layout of PyTorch's NCHW);
    vision transformers also accept `"NLC"`, for `(C, L, N)` tokens.

Returns `(features, intermediates)`, where `features` is the output of
[`forward_features`](@ref) (truncated with `stop_early`), or only `intermediates` with
`intermediates_only = true`. Vision transformers take `return_prefix_tokens = true`, which
returns each intermediate as a `(spatial, prefix)` tuple, where `prefix` holds the class and
register tokens.
"""
function forward_intermediates end

"""
    feature_info(model) -> Vector{NamedTuple}

The intermediates of [`forward_intermediates`](@ref), as timm's `model.feature_info`: for each,
its number of channels `num_chs`, its `reduction` (stride) relative to the input, and the
PyTorch name `module_name` of the module that outputs it (timm's `module`). For a language
model, those of its decoder layers: `hidden_size` channels, a reduction of 1.
"""
function feature_info end
