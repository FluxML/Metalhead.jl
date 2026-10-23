const ELTYPE_CONVERTERS = Dict{DataType, Function}(Float16 => f16, Float32 => f32, Float64 =>
    f64)

nt(d::AbstractDict) = (; (Symbol(k) => jsonvalue(v) for (k, v) in d)...)
jsonvalue(v::AbstractVector) = Tuple(jsonvalue.(v))
jsonvalue(v::AbstractDict) = nt(v)
jsonvalue(v) = v

# A model source: a Hub repository, or a local directory laid out like one.
struct ModelSource
    repo::Union{Nothing, String}
    dir::Union{Nothing, String}
    revision::String
end

function source_file(s::ModelSource, file)
    return s.dir === nothing ? hf_download(s.repo, file; revision = s.revision) :
           joinpath(s.dir, file)
end
function source_weights(s::ModelSource)
    return s.dir === nothing ? hf_weights(s.repo; revision = s.revision) :
           [joinpath(s.dir, "model.safetensors")]
end

function bare_spec(name)
    haskey(REGISTRY, name) && return REGISTRY[name]
    return error("`$name` is neither a ported architecture (see `list_models()`), a Hub repository (`org/name`) nor a directory")
end

# What `name` stands for: a local directory, a Hub repository, or the repository of the
# default checkpoint of a bare architecture. A validated repository defaults to the commit it
# was validated at.
function model_source(name::AbstractString, revision)
    isdir(name) && return ModelSource(nothing, abspath(name), "")
    if occursin('/', name)
        pinned = [sha for spec in values(REGISTRY) for (repo, sha) in spec.references
                  if repo == name]
        return ModelSource(String(name), nothing,
            something(revision, isempty(pinned) ? "main" : first(pinned)))
    end
    spec = bare_spec(name)
    isempty(spec.references) &&
        error("`$name` has no validated pretrained checkpoint; pass a Hub repository instead")
    repo, sha = first(spec.references)
    return ModelSource(repo, nothing, something(revision, sha))
end

"""
    create_model(name; pretrained = false, revision = nothing, eltype = Float32, kwargs...)

Build a ported model.

`name` is a Hub repository (`"timm/resnet50.a1_in1k"`), a bare architecture (`"resnet50"`; see
[`list_models`](@ref)), or a local directory containing a `config.json` and `model.safetensors`.

By default the weights are random and nothing is downloaded, except the `config.json` of a
repository or directory. With `pretrained = true` they are loaded from the repository (for a
bare architecture, from its default validated checkpoint). `revision` selects a branch, tag or
commit of the repository; validated checkpoints default to the commit they were validated at.
The remaining keyword arguments are the original library's model arguments, e.g. timm's
`num_classes`. Changing `num_classes` of a pretrained model re-initialises its classifier.

The model is returned in test mode; call `Flux.trainmode!` before training.
"""
function create_model(name::AbstractString; pretrained::Bool = false,
        revision::Union{Nothing, AbstractString} = nothing,
        eltype::Type{<:AbstractFloat} = Float32, kwargs...)
    if !pretrained && !isdir(name) && !occursin('/', name)
        spec = bare_spec(name)
        model = spec.build(; merge(spec.defaults, values(kwargs))...)
        return Flux.testmode!(ELTYPE_CONVERTERS[eltype](model))
    end
    source = model_source(name, revision)
    config = JSON.parsefile(source_file(source, "config.json"))
    format, architecture = identify(config)
    spec = lookup(architecture)
    args = model_args(format, spec, config, NamedTuple(kwargs))
    model = ELTYPE_CONVERTERS[eltype](spec.build(; args...))
    if pretrained
        skip = skip_prefix(format, config, args)
        load_state!(model, Checkpoint(source_weights(source)); skip, ignore = [DEFAULT_IGNORE;
                                                                               spec.ignore])
    end
    return Flux.testmode!(model)
end

"""
    forward_features(model, x)

Run `model` up to its last feature map, before pooling and classification (timm's `forward_features`).

This and [`forward_head`](@ref), [`forward_intermediates`](@ref) and [`feature_info`](@ref) are
the interface of image backbones and classifiers, not of every ported model.
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
    transformers also accept `"NLC"`, for `(C, L, N)` tokens.

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
PyTorch name `module_name` of the module that outputs it (timm's `module`).
"""
function feature_info end
