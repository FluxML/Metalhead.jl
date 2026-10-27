# The modules.json of a sentence-transformers model: the modules of its pipeline, in order,
# each `{"idx", "name", "path", "type"}`, whose files are in the subfolder `path` ("" for
# the root, where the Transformer module keeps the config.json and weights of its
# transformers model). `source_config` gives it as `Dict("modules.json" => modules)`, ahead
# of that config.json.
struct SentenceTransformersFormat <: AbstractConfigFormat end
const SENTENCE_TRANSFORMERS = SentenceTransformersFormat()
Base.show(io::IO, ::SentenceTransformersFormat) = print(io, "sentence-transformers")

function architecture_of(::SentenceTransformersFormat, config)
    return haskey(config, "modules.json") ? "SentenceTransformer" : nothing
end

# The settings of a module of class `T` from the configuration `dict` of `file`: the values
# of the keys of `defaults` (`name = default`) over the defaults, the former names of
# `renamed` (`old = :new`) read as their new ones; the keys of `ignored` (`key = "reason"`)
# are skipped and any other key warns once, as for processors.
function read_settings(T, dict, file; defaults::NamedTuple, renamed::NamedTuple = (;),
        ignored::NamedTuple = (;))
    values = Dict{Symbol, Any}(pairs(defaults))
    for (k, v) in something(dict, Dict{String, Any}())
        key = get(renamed, Symbol(k), Symbol(k))
        if haskey(defaults, key)
            values[key] = jsonvalue(v)
        elseif !haskey(ignored, key)
            warn_unknown(T, key, file)
        end
    end
    return NamedTuple{keys(defaults)}(map(k -> values[k], keys(defaults)))
end

# The error of a setting whose value the port doesn't support.
function unsupported(key, value, file, supported)
    return throw(ArgumentError("`$key = $(repr(value))` in $file is not supported by Metalhead; supported: $supported"))
end

# The JSON file `name` of a source, parsed, or `nothing` if the source doesn't have it.
function read_file(s::ModelSource, name)
    path = source_file_if_exists(s, name)
    return path === nothing ? nothing : JSON.parsefile(path)
end

# The class of a modules.json `type`: the class name of a sentence-transformers module, such
# as `sentence_transformers.models.Pooling` (sentence-transformers < 5) or
# `sentence_transformers.sentence_transformer.modules.pooling.Pooling`.
function module_class(type)
    parts = split(type, '.')
    first(parts) == "sentence_transformers" ||
        error("the module `$type` of modules.json is remote code, which Metalhead doesn't run")
    class = String(last(parts))
    haskey(MODULE_LOADERS, class) ||
        error("the module `$type` of modules.json has not been ported to Metalhead (ported: $(join(sort!(collect(keys(MODULE_LOADERS))), ", ")))")
    return class
end

# The loaders of the ported modules, by class name: `load(source; pretrained, eltype)`
# builds the module whose files are in `source`, the subfolder of its modules.json entry.
const MODULE_LOADERS = Dict{String, Any}()

const MODULES_JSON_IGNORED = (;
    idx = "is the position of the module, as the order of modules.json is",
    name = "names the module in the PyTorch model")

const MODEL_CONFIG_IGNORED = (;
    __version__ = "records the versions of the libraries that saved the model",
    requirements = "lists the versions of Python packages that the model needs")

# The settings of the model in config_sentence_transformers.json (absent from models saved
# before sentence-transformers 2), which sentence-transformers' `SentenceTransformer` reads.
function model_config(s::ModelSource)
    file = "config_sentence_transformers.json"
    c = read_settings(SentenceTransformer, read_file(s, file), file;
        defaults = (; model_type = "SentenceTransformer", prompts = (;),
            default_prompt_name = nothing, similarity_fn_name = nothing,
            truncate_dim = nothing),
        ignored = MODEL_CONFIG_IGNORED)
    c.model_type == "SentenceTransformer" ||
        unsupported("model_type", c.model_type, file, "\"SentenceTransformer\"")
    return Base.structdiff(c, NamedTuple{(:model_type,)})
end

# A sentence-transformers model is built from its modules, each loaded from its subfolder
# with the same `pretrained` and `eltype`; the keyword arguments override the settings of
# config_sentence_transformers.json, as those of sentence-transformers'
# `SentenceTransformer`.
function instantiate(::SentenceTransformersFormat, spec, source, config; pretrained::Bool,
        eltype, kwargs...)
    entries = config["modules.json"]
    entries isa AbstractVector && !isempty(entries) ||
        error("modules.json lists no modules")
    modules = map(entries) do entry
        e = read_settings(SentenceTransformer, entry, "modules.json";
            defaults = (; type = nothing, path = nothing, kwargs = ()),
            ignored = MODULES_JSON_IGNORED)
        e.type isa AbstractString && e.path isa AbstractString ||
            error("a module of modules.json has no `type` or `path`")
        isempty(e.kwargs) ||
            unsupported("kwargs", e.kwargs, "modules.json", "none, as no ported module takes keyword arguments")
        load = MODULE_LOADERS[module_class(e.type)]
        return load(subsource(source, e.path); pretrained, eltype)
    end
    settings = model_config(source)
    known = keys(settings)
    for k in keys(kwargs)
        k in known ||
            throw(ArgumentError("`$k` is not a setting of a SentenceTransformer; its settings are $(join(("`$s`" for s in known), ", "))"))
    end
    return spec.build(; modules, merge_settings(settings, NamedTuple(kwargs))...)
end

# The settings of config_sentence_transformers.json with the overrides of the user: prompts
# are merged, those of the user taking precedence, as sentence-transformers'
# `_parse_model_config`.
function merge_settings(settings, overrides)
    merged = merge(settings, overrides)
    haskey(overrides, :prompts) || return merged
    prompts = merge(settings.prompts, (;
        (Symbol(k) => v for (k, v) in pairs(overrides.prompts))...))
    return merge(merged, (; prompts))
end

# A SentenceTransformer prepares its inputs itself, with the tokenizer of its Transformer
# module: it has no processor.
function instantiate_processor(::SentenceTransformersFormat, files, config; kwargs...)
    return error("a sentence-transformers model has no processor: it prepares its inputs itself, with the tokenizer of its Transformer module. Use `encode(model, texts)` (with `using HuggingFaceTokenizers`) or `Metalhead.SentenceTransformers.preprocess(model, texts)`, where `model = create_model(repo)`")
end
