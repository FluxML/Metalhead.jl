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

function library_of(config)
    return haskey(config, "architecture") ? :timm :
           haskey(config, "model_type") ? :transformers :
           error("cannot tell which library this config.json is for")
end

"""
    create_model(name; pretrained = false, revision = nothing, eltype = Float32, kwargs...)

Build a ported model.

`name` is a Hub repository (`"timm/resnet50.a1_in1k"`), a timm model name with its pretrained
tag (`"resnet50.a1_in1k"`), a bare architecture (`"resnet50"`), or a local directory containing
a `config.json` and `model.safetensors`.

With `pretrained = true` the weights are loaded from the repository (for a bare architecture,
from its default validated checkpoint). `revision` selects a branch, tag or commit of the
repository; validated checkpoints default to the commit they were validated at.
The remaining keyword arguments are the original library's model arguments, e.g. `num_classes`.
Changing `num_classes` of a pretrained model re-initialises its classifier.

The model is returned in test mode; call `Flux.trainmode!` before training.
"""
function create_model(name::AbstractString; pretrained::Bool = false,
        revision::Union{Nothing, AbstractString} = nothing,
        eltype::Type{<:AbstractFloat} = Float32, kwargs...)
    if isdir(name)
        source = ModelSource(nothing, abspath(name), "")
    elseif !occursin('/', name) && !occursin('.', name)
        spec = lookup(:timm, name)
        if !pretrained
            model = spec.build(; merge(spec.defaults, values(kwargs))...)
            return Flux.testmode!(ELTYPE_CONVERTERS[eltype](model))
        end
        isempty(spec.references) &&
            error("`$name` has no validated pretrained checkpoint; pass a Hub repository instead")
        repo, sha = first(spec.references)
        source = ModelSource(repo, nothing, something(revision, sha))
    else
        repo = occursin('/', name) ? String(name) : "timm/$name"
        pinned = [sha for spec in values(REGISTRY)
                  for (r, sha) in spec.references if r == repo]
        source = ModelSource(repo, nothing, something(revision, isempty(pinned) ? "main" :
                                                                first(pinned)))
    end
    config = JSON.parsefile(source_file(source, "config.json"))
    library = library_of(config)
    library == :timm || error("$library models are not supported yet")
    spec = lookup(library, config["architecture"])
    model_args = nt(get(config, "model_args", Dict()))
    ckpt_classes = get(config, "num_classes", nothing)
    from_config = ckpt_classes === nothing ? (;) : (; num_classes = ckpt_classes)
    cfg = get(config, "pretrained_cfg", Dict())
    # timm's _update_default_model_kwargs: models of a fixed input size are built for it
    from_cfg = get(cfg, "fixed_input_size", false) && haskey(cfg, "input_size") ?
               (; img_size = Tuple(cfg["input_size"][2:3])) : (;)
    args = merge(spec.defaults, from_cfg, model_args, from_config, values(kwargs))
    model = ELTYPE_CONVERTERS[eltype](spec.build(; args...))
    if pretrained
        if haskey(args, :in_chans) &&
           args.in_chans != get(get(cfg, "input_size", [3]), 1, 3)
            error("loading pretrained weights with a different `in_chans` is not supported yet")
        end
        skip = nothing
        if ckpt_classes !== nothing && args.num_classes != ckpt_classes
            skip = get(cfg, "classifier", nothing)
            skip === nothing &&
                error("config.json does not name the classifier; cannot change `num_classes`")
        end
        load_state!(model, Checkpoint(source_weights(source)); skip, ignore = [DEFAULT_IGNORE;
                                                                               spec.ignore])
    end
    return Flux.testmode!(model)
end

"""
    pretrained_cfg(name; revision = nothing) -> NamedTuple

Preprocessing settings of a pretrained checkpoint (`mean`, `std`, `input_size`, `crop_pct`,
`interpolation`, ...), as stored in its `config.json`. `input_size` is in PyTorch order (C, H, W).
"""
function pretrained_cfg(name::AbstractString; revision::Union{Nothing, AbstractString} = nothing)
    source = isdir(name) ? ModelSource(nothing, abspath(name), "") :
             ModelSource(occursin('/', name) ? String(name) : "timm/$name",
        nothing, something(revision, "main"))
    return nt(get(JSON.parsefile(source_file(source, "config.json")), "pretrained_cfg", Dict()))
end

"""
    forward_features(model, x)

Run `model` up to its last feature map, before pooling and classification (timm's `forward_features`).
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
