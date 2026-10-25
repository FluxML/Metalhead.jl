# Processors: what turns the raw inputs of a checkpoint into the inputs of its model, and
# the model's outputs back into results, configured by the files of its repository. The
# classes are ported next to their models; this file has `create_processor`, the registry
# of classes, transformers' AutoProcessor lookup of a repository's class, and the strict
# reading of settings: a configuration over the class defaults, overrides, and where each
# setting comes from.

"""
    Processor

A processor: an image processor, a tokenizer, or a processor that combines them, as
[`create_processor`](@ref) returns them. Each is a plain immutable struct whose fields are
the attributes of the Python class (its settings, and the processors it combines), with a
keyword constructor whose defaults are those of the Python class, and a `provenance` field
that records where each setting comes from, for `show`.
"""
abstract type Processor end

"""
    ImageProcessor <: Processor

A processor of images, called as `ip(images)` on one image or a vector of them (see
[`image_array`](@ref Metalhead.image_array)).
"""
abstract type ImageProcessor <: Processor end

# Where a setting comes from: `source` is the file that sets it, "class default" or
# "override", and `value` its value then; an override keeps the origin it replaced.
struct Origin
    source::String
    value::Any
    replaced::Union{Nothing, Origin}
end
Origin(source, value) = Origin(source, value, nothing)

"""
    processor_components(T) -> NamedTuple

The components of the processor class `T`, as `name = type`: the fields that hold the
processors it combines, loaded from the same repository, e.g.
`(; image_processor = Sam3ImageProcessor, tokenizer = TokenizersBackend)`. Empty (the
default) for an image processor or a tokenizer. A component is loaded by
[`load_processor`](@ref Metalhead.load_processor) of its type, and the fields of `T` that
aren't components (nor `provenance`) are its settings.
"""
processor_components(::Type) = (;)

"""
    ignored_keys(T) -> NamedTuple

The keys of a configuration that the processor class `T` knows but doesn't read, as
`key = "reason"`, e.g. `(; mask_size = "only resizes the segmentation maps of training")`.
Other keys that aren't settings of `T` warn once. An override of an ignored key is an error
that gives the reason.
"""
ignored_keys(::Type) = (;)

"""
    restricted_keys(T) -> NamedTuple

The keys of features that the processor class `T` doesn't port, as `key = allowed values`,
e.g. `(; do_center_crop = (false,))`: another value, in a file or an override, is an error.
`null` in a file is missing, as for any key.
"""
restricted_keys(::Type) = (;)

"""
    legacy_keys(T) -> NamedTuple

Former names of settings of the processor class `T`, as `key = path`, where `path` is the
name of a setting or the path to an entry of a nested one, e.g.
`(; min_pixels = (:size, :shortest_edge))`. The value of a legacy key replaces the one at
its path, in files and overrides, and drops the entry of the legacy key from the nested
setting, as transformers does.
"""
legacy_keys(::Type) = (;)

"""
    processor_fields(T, args::Tuple) -> Tuple

The values of the fields of a processor of class `T` from `args`, the arguments of its
constructor in the order of the fields: the forms of a setting that the Python library also
takes, or that read naturally, converted to the type of its field (an integer `size` to
`(; height, width)`), and an `ArgumentError` naming the forms of a setting for another
value, rather than `convert`'s `MethodError`. `args` as they are by default, which `new`
converts to the types of the fields.
"""
processor_fields(::Type, args::Tuple) = args

"""
    check_processor(p)

Throw if a setting of the processor `p` has a value that its port doesn't implement, naming
the supported values, or if a component isn't of its class (see
[`processor_components`](@ref Metalhead.processor_components)). The constructors of the
processor classes call it, so that a processor built by hand fails early too, and so does
`create_processor`.
"""
check_processor(p) = check_components(p)

function check_components(p)
    for (name, T) in pairs(processor_components(typeof(p)))
        c = getfield(p, name)
        c === nothing || c isa T ||
            error("the `$name` of $(nameof(typeof(p))) is a $(nameof(T)) (or `nothing`), got a $(typeof(c))")
    end
    return nothing
end

"""
    @processor struct T <: Processor ... end
    @processor (components...,) struct T <: Processor ... end

`Base.@kwdef` for a processor class, whose constructors make the processor from
[`processor_fields`](@ref Metalhead.processor_fields) of their arguments, and then run
[`check_processor`](@ref Metalhead.check_processor) on it. With the names of its
components, `T(components...; kwargs...)` also makes it from them, e.g.
`Sam3Processor(image_processor, tokenizer; point_pad_value = -10)`.
"""
macro processor(args...)
    def = last(args)
    components = length(args) == 2 ? first(args).args : ()
    Meta.isexpr(def, :struct) || error("@processor takes a struct definition")
    name = Meta.isexpr(def.args[2], :<:) ? def.args[2].args[1] : def.args[2]
    fields = Symbol[]
    for line in def.args[3].args
        field = Meta.isexpr(line, :(=)) ? line.args[1] : line
        field = Meta.isexpr(field, :(::)) ? field.args[1] : field
        field isa Symbol && push!(fields, field)
    end
    push!(def.args[3].args, :(function $name($(fields...))
        p = new($processor_fields($name, ($(fields...),))...)
        $check_processor(p)
        return p
    end))
    out = Expr(:block, :(Base.@kwdef $def))
    if !isempty(components)
        call = Expr(:call, name, Expr(:parameters, components..., :(kwargs...)))
        push!(out.args, :($name($(components...); kwargs...) = $call))
    end
    return esc(out)
end

# The settings of a class, as `name = default`: the fields of its default instance that
# aren't components.
setting_names(::Type{T}) where {T} = Tuple(k for k in fieldnames(T)
                                           if k !== :provenance &&
                                              !haskey(processor_components(T), k))
default_settings(::Type{T}) where {T <: Processor} = settings(T())
settings(p) = NamedTuple{setting_names(typeof(p))}(map(k -> getfield(p, k),
    setting_names(typeof(p))))
function components(p)
    names = keys(processor_components(typeof(p)))
    return NamedTuple{names}(map(k -> getfield(p, k), names))
end
provenance(p) = hasfield(typeof(p), :provenance) ? getfield(p, :provenance) : (;)

"""
    merge_nested(base, override)

`override` merged into `base`: entry by entry, recursively, where both are NamedTuples, so
that `merge_nested((; a = 1, b = (; c = 2, d = 3)), (; b = (; d = 4)))` is
`(; a = 1, b = (; c = 2, d = 4))`; else `override`.
"""
function merge_nested(base::NamedTuple, override::NamedTuple)
    merged = map(k -> k => merge_nested(get(base, k, nothing), override[k]), keys(override))
    return merge(base, (; merged...))
end
merge_nested(base, override) = override

# The NamedTuple that sets the entry at `path` of a nested setting to `value`.
nested_at(path::Tuple{}, value) = value
function nested_at(path::Tuple, value)
    return NamedTuple{(first(path),)}((nested_at(Base.tail(path), value),))
end

# A nested setting without the entry of the legacy key `key`, which transformers drops when
# the legacy key sets its new entry (Qwen2-VL's `size["min_pixels"]`).
drop_legacy(setting::NamedTuple, key) = Base.structdiff(setting, NamedTuple{(key,)})
drop_legacy(setting, key) = setting

# Keys that every configuration may have, which name classes rather than set them.
const SERVICE_KEYS = (; processor_class = "names the class of the processor",
    image_processor_type = "names the class of the image processor",
    video_processor_type = "names the class of the video processor",
    feature_extractor_type = "names the class of the feature extractor",
    auto_map = "names remote code, which Metalhead doesn't run")

const WARNED_KEYS = Set{Tuple{Any, Symbol}}()
const WARNED_LOCK = ReentrantLock()

function warn_unknown(T, key, file)
    first_time = lock(WARNED_LOCK) do
        (T, key) in WARNED_KEYS && return false
        push!(WARNED_KEYS, (T, key))
        return true
    end
    first_time && @warn "`$key` in $file is not a setting of $(nameof(T)); it is ignored"
    return nothing
end

function check_restricted(T, key, value, where)
    allowed = restricted_keys(T)[key]
    any(isequal(value), allowed) && return nothing
    return error("`$key = $(repr(value))`$where is not supported by $(nameof(T)) (supported: $(join(repr.(allowed), ", ")))")
end

# The names that an override of a setting of `T` may take, components excluded.
own_override_names(T) = (keys(default_settings(T))..., keys(legacy_keys(T))...,
    keys(restricted_keys(T))...)

function edit_distance(a::AbstractString, b::AbstractString)
    a, b = collect(a), collect(b)
    d = collect(0:length(b))
    for i in eachindex(a)
        previous, d[1] = d[1], i
        for j in eachindex(b)
            previous, d[j + 1] = d[j + 1],
            min(d[j + 1] + 1, d[j] + 1, previous + (a[i] != b[j]))
        end
    end
    return d[end]
end

# " (`Class` has `name`)" for the setting of `classes` closest to `key`, if one is close.
function close_match(key, classes)
    best = nothing
    for T in classes, name in own_override_names(T)
        d = edit_distance(string(key), string(name))
        d <= 2 && (best === nothing || d < best[1]) && (best = (d, T, name))
    end
    best === nothing && return ""
    return " ($(nameof(best[2])) has `$(best[3])`)"
end

function ignored_reason(key, classes)
    for T in classes
        haskey(ignored_keys(T), key) &&
            return "`$key` is not used by $(nameof(T)): $(ignored_keys(T)[key])"
    end
    return nothing
end

"""
    processor_from_entries(T, entries; components = (;), kwargs...) -> T

The processor of class `T` with the settings of `entries`, `(key, value, file)` triples in
the order the files are read (a later entry replaces an earlier one), over the defaults of
the class, and then with the overrides `kwargs` merged into them with
[`merge_nested`](@ref Metalhead.merge_nested). Values are Julia values (see `jsonvalue`);
`nothing` is missing, and `Some(nothing)` sets `nothing` (a `null` special token of a
tokenizer, which replaces the default of its class). `components` are passed to the
constructor as they are.

The keys of `entries` are checked strictly: a legacy key sets the setting at its path, a
restricted key must have an allowed value, an ignored or a service key is skipped, and any
other key warns once. An override that isn't a setting is an error.
"""
function processor_from_entries(::Type{T}, entries; components::NamedTuple = (;),
        kwargs...) where {T}
    defaults = default_settings(T)
    values = Dict{Symbol, Any}()
    origins = Dict{Symbol, Origin}()
    legacy = Tuple{Symbol, Any, String}[]
    for (k, v, file) in entries
        v === nothing && continue
        v isa Some && (v = something(v))
        key = Symbol(k)
        if haskey(defaults, key)
            values[key] = v
            origins[key] = Origin(file, v)
        elseif haskey(legacy_keys(T), key)
            push!(legacy, (key, v, file))
        elseif haskey(restricted_keys(T), key)
            check_restricted(T, key, v, " in $file")
        elseif !haskey(ignored_keys(T), key) && !haskey(SERVICE_KEYS, key)
            warn_unknown(T, key, file)
        end
    end
    legacy_names = Dict{Symbol, Vector{String}}()
    for (key, v, file) in legacy
        path = legacy_keys(T)[key]
        name = first(path)
        values[name] = merge_nested(drop_legacy(get(values, name, defaults[name]), key),
            nested_at(Base.tail(path), v))
        former = push!(get!(legacy_names, name, String[]), "`$key`")
        origins[name] = Origin("$file ($(join(former, ", ")))", values[name])
    end
    for (key, v) in pairs(NamedTuple(kwargs))
        if haskey(legacy_keys(T), key)
            path = legacy_keys(T)[key]
            haskey(values, first(path)) &&
                (values[first(path)] = drop_legacy(values[first(path)], key))
            key, v = first(path), nested_at(Base.tail(path), v)
        end
        if !haskey(defaults, key)
            if haskey(restricted_keys(T), key)
                check_restricted(T, key, v, "")
                continue
            end
            reason = ignored_reason(key, (T,))
            reason === nothing || error(reason)
            error("$(nameof(T)) has no setting `$key`$(close_match(key, (T,)))")
        end
        replaced = get(origins, key, Origin("class default", defaults[key]))
        values[key] = merge_nested(get(values, key, defaults[key]), v)
        origins[key] = Origin("override", values[key], replaced)
    end
    p = T(; components..., (k => values[k] for k in keys(defaults) if haskey(values, k))...)
    # the values as the constructor stored them, converted to the types of its fields
    stored = settings(p)
    prov = map(keys(defaults)) do k
        o = get(origins, k, Origin("class default", nothing))
        return k => Origin(o.source, stored[k], o.replaced)
    end
    p = T(; components..., stored..., provenance = (; prov...))
    check_processor(p)
    return p
end

# A dictionary of a JSON file as entries of `processor_from_entries`, in Julia values.
json_entries(dict, file) = [(String(k), jsonvalue(v), file) for (k, v) in dict]

"""
    ProcessorFiles(source)

The files of a processor's source (a repository at a revision, or a directory, see
`ModelSource`), each JSON file read once with `read_json(files, name)` (`nothing` if the
source doesn't have it). A branch or a tag is pinned to its commit when the first file is
found, as transformers' `resolve_revision` pins it: the files of one processor come from
one commit, and the files it downloads later (tokenizer.json) are then found in the cache
without asking the Hub.
"""
mutable struct ProcessorFiles
    source::ModelSource
    const json::Dict{String, Any}
end
ProcessorFiles(source::ModelSource) = ProcessorFiles(source, Dict{String, Any}())

function read_json(files::ProcessorFiles, name)
    return get!(files.json, name) do
        path = source_file_if_exists(files, name)
        return path === nothing ? nothing : JSON.parsefile(path)
    end
end
function has_file(files::ProcessorFiles, name)
    haskey(files.json, name) && return files.json[name] !== nothing
    return source_file_if_exists(files, name) !== nothing
end

function source_file_if_exists(files::ProcessorFiles, name)
    path = source_file_if_exists(files.source, name)
    path === nothing || (files.source = pinned(files.source, name, path))
    return path
end

is_commit(revision) = occursin(r"^[0-9a-f]{40}$", revision)

# `s` at the commit of the snapshot of the cache that holds `path`, its `file`.
function pinned(s::ModelSource, file, path)
    s.dir === nothing && !is_commit(s.revision) || return s
    snapshot = path
    for _ in split(join_path(s.subfolder, file), '/')
        snapshot = dirname(snapshot)
    end
    sha = basename(snapshot)
    is_commit(sha) && basename(dirname(snapshot)) == "snapshots" || return s
    return ModelSource(s.repo, s.dir, sha, s.subfolder)
end

# Files that a repository lacks, so that each is asked of the Hub once per session. A file
# that is only absent from the cache, offline or without network, isn't recorded.
const MISSING_FILES = Set{Tuple{String, String, String}}()
const MISSING_LOCK = ReentrantLock()

# The local path of `file` of a source, or `nothing` if it doesn't have it. Offline or
# without network, a file that isn't in the cache counts as missing, as in
# `hf_download_first`.
function source_file_if_exists(s::ModelSource, file)
    file = join_path(s.subfolder, file)
    if s.dir !== nothing
        path = joinpath(s.dir, file)
        return isfile(path) ? path : nothing
    end
    key = (s.repo, s.revision, file)
    lock(() -> key in MISSING_FILES, MISSING_LOCK) && return nothing
    try
        return hf_download(s.repo, file; revision = s.revision)
    catch e
        if is_not_found(e)
            lock(() -> push!(MISSING_FILES, key), MISSING_LOCK)
            return nothing
        end
        is_uncached(e, s.repo, file, s.revision) && return nothing
        rethrow()
    end
end

"""
    SourceFile(source, file)

A file of a source, downloaded when [`local_path`](@ref Metalhead.local_path) first needs
it: the `tokenizer.json` of a tokenizer, which only strings need.
"""
struct SourceFile
    source::ModelSource
    file::String
end

"""
    local_path(file) -> String

The local path of a `SourceFile`, downloading it, or of a path.
"""
local_path(path::AbstractString) = path
function local_path(f::SourceFile)
    s = f.source
    if s.dir !== nothing
        path = source_file_if_exists(s, f.file)
        path === nothing && error("$(source_name(s)) has no $(f.file)")
        return path
    end
    file = join_path(s.subfolder, f.file)
    return last(hf_download_first(s.repo, [file], f.file; revision = s.revision))
end

# The local path of a `SourceFile`, or `nothing` if its source lacks it or it isn't in the
# cache, offline or without network.
local_path_if_exists(f::SourceFile) = source_file_if_exists(f.source, f.file)
local_path_if_exists(path::AbstractString) = isfile(path) ? path : nothing

function source_name(s::ModelSource)
    s.dir === nothing || return joinpath(s.dir, s.subfolder)
    return string(join_path(s.repo, s.subfolder), "@", s.revision)
end
Base.show(io::IO, f::SourceFile) = print(io, join_path(source_name(f.source), f.file))

# The processor classes, by the names of the Python classes that configurations name.
const PROCESSOR_CLASSES = Dict{String, Type}()
# The processor class of a model type, AutoProcessor's last resort.
const PROCESSOR_MODEL_TYPES = Dict{String, String}()

"""
    register_processor!(name, T; aliases = (), model_types = ())

Register the processor class `T` as the port of transformers' (or another library's)
class `name`, which configurations name in `processor_class` or `image_processor_type`, and
of the classes `aliases`. The suffixes `Fast` and `Pil` of class names, those of
transformers' former image processors and of its PIL backend, are dropped before lookup.
`model_types` are the `model_type`s of config.json whose processor is `T` (AutoProcessor's
`PROCESSOR_MAPPING`), for repositories whose processor files don't name a class.
"""
function register_processor!(name::AbstractString, T::Type; aliases = (), model_types = ())
    for n in (name, aliases...)
        old = get(PROCESSOR_CLASSES, n, T)
        old === T || error("processor class `$n` is already registered as $old")
        PROCESSOR_CLASSES[n] = T
    end
    for m in model_types
        PROCESSOR_MODEL_TYPES[m] = name
    end
    return T
end

class_name(name) = replace(name, r"(Fast|Pil)$" => "")

function processor_type(name, where)
    T = get(PROCESSOR_CLASSES, class_name(name), nothing)
    T === nothing || return T
    ported = join(sort!(collect(keys(PROCESSOR_CLASSES))), ", ")
    return error("the processor class `$name` ($where) has not been ported to Metalhead (ported: $ported)")
end

# The class of a source's processor, and where it is named, as AutoProcessor finds it: the
# `processor_class` of processor_config.json, preprocessor_config.json (or
# video_preprocessor_config.json without it), tokenizer_config.json or config.json, the
# processor of the model type, else a tokenizer, else an image processor.
function auto_processor_class(files::ProcessorFiles, config)
    candidates = ["processor_config.json",
        has_file(files, "preprocessor_config.json") ? "preprocessor_config.json" :
        "video_preprocessor_config.json", "tokenizer_config.json"]
    for file in candidates
        dict = read_json(files, file)
        class = dict === nothing ? nothing : get(dict, "processor_class", nothing)
        class === nothing || return processor_type(class, "`processor_class` in $file")
    end
    if config !== nothing
        class = get(config, "processor_class", nothing)
        class === nothing ||
            return processor_type(class, "`processor_class` in config.json")
        model_type = get(config, "model_type", nothing)
        haskey(PROCESSOR_MODEL_TYPES, model_type) &&
            return processor_type(PROCESSOR_MODEL_TYPES[model_type],
                "the processor of the model type `$model_type`")
    end
    if has_file(files, "tokenizer_config.json") || has_file(files, "tokenizer.json")
        return processor_type("TokenizersBackend", "a tokenizer")
    end
    found = image_processor_config(files)
    if found !== nothing
        dict, file = found
        class = get(dict, "image_processor_type", nothing)
        class === nothing &&
            error("the image processor of $file has no `image_processor_type`")
        return processor_type(class, "`image_processor_type` in $file")
    end
    return error("$(source_name(files.source)) has no processor files (processor_config.json, preprocessor_config.json, tokenizer_config.json or tokenizer.json)")
end

# The configuration of an image processor and its file: the `image_processor` of
# processor_config.json, which wins as in transformers, else preprocessor_config.json.
function image_processor_config(files::ProcessorFiles)
    processor = read_json(files, "processor_config.json")
    processor !== nothing && haskey(processor, "image_processor") &&
        return processor["image_processor"], "processor_config.json"
    dict = read_json(files, "preprocessor_config.json")
    return dict === nothing ? nothing : (dict, "preprocessor_config.json")
end

"""
    load_processor(T, files::ProcessorFiles; kwargs...) -> T

The processor of class `T` from the files of a repository, with the overrides `kwargs`.
An image processor reads the `image_processor` of processor_config.json, else
preprocessor_config.json. A processor with [`processor_components`](@ref
Metalhead.processor_components) loads each component with its own method, routing each
override to the one component that has a setting of that name, and reads its own settings
from processor_config.json. A tokenizer has a method of its own.
"""
function load_processor(::Type{T}, files::ProcessorFiles;
        kwargs...) where {T <: ImageProcessor}
    found = image_processor_config(files)
    found === nothing &&
        error("$(source_name(files.source)) has no image processor (preprocessor_config.json, or `image_processor` in processor_config.json)")
    dict, file = found
    class = get(dict, "image_processor_type", nothing)
    class === nothing || processor_type(class, "`image_processor_type` in $file") === T ||
        error("the image processor of $file is a $class, not a $(nameof(T))")
    return processor_from_entries(T, json_entries(dict, file); kwargs...)
end

function load_processor(::Type{T}, files::ProcessorFiles; kwargs...) where {T <: Processor}
    parts = processor_components(T)
    isempty(parts) && error("$(nameof(T)) has no `load_processor` method")
    routed = route_overrides(T, NamedTuple(kwargs))
    loaded = map(keys(parts)) do name
        return name => load_processor(parts[name], files; routed[name]...)
    end
    dict = something(read_json(files, "processor_config.json"), Dict{String, Any}())
    entries = [(String(k), jsonvalue(v), "processor_config.json")
               for (k, v) in dict if !haskey(parts, Symbol(k))]
    return processor_from_entries(T, entries; components = (; loaded...), routed.self...)
end

# The overrides of each component of `T` (and of `T` itself, `self`): each to the one that
# has a setting of its name.
function route_overrides(::Type{T}, kwargs::NamedTuple) where {T}
    parts = processor_components(T)
    accepts(C, key) = key in own_override_names(C) ||
                      any(S -> accepts(S, key), values(processor_components(C)))
    routed = Dict(name => Pair{Symbol, Any}[] for name in (:self, keys(parts)...))
    for (key, v) in pairs(kwargs)
        owners = [name for name in keys(parts) if accepts(parts[name], key)]
        key in own_override_names(T) && push!(owners, :self)
        if isempty(owners)
            classes = (T, values(parts)...)
            reason = ignored_reason(key, classes)
            reason === nothing || error(reason)
            error("no component of $(nameof(T)) has a setting `$key`$(close_match(key, classes))")
        end
        length(owners) == 1 ||
            error("`$key` is a setting of several components of $(nameof(T)) ($(join(owners, ", "))); set it on one of them after loading")
        push!(routed[only(owners)], key => v)
    end
    return (; (name => (; routed[name]...) for name in keys(routed))...)
end

"""
    check_inputs(model, inputs::NamedTuple, required; hint = "")

Throw if the NamedTuple `inputs` of `model(inputs)` lacks keys of `required`, naming them
and the keys it has, so that the inputs of another model fail at the call rather than deep
inside. Every `model(inputs::NamedTuple)` method calls it on the keys it can't do without.
"""
function check_inputs(model, inputs::NamedTuple, required; hint::AbstractString = "")
    absent = filter(k -> !haskey(inputs, k), collect(Symbol, required))
    isempty(absent) && return nothing
    names(ks) = join(("`$k`" for k in ks), ", ", " and ")
    return throw(ArgumentError("$(nameof(typeof(model))) needs the inputs $(names(required)), but `inputs` has no $(names(absent)) (it has $(isempty(inputs) ? "no keys" : names(keys(inputs))))$(isempty(hint) ? "" : "; $hint")"))
end

"""
    instantiate_processor(format::ConfigFormat, files::ProcessorFiles, config; kwargs...)
        -> Union{Nothing, Processor}

The processor of a repository whose config.json, parsed as `config`, is in `format` and
carries the preprocessing of its checkpoint (timm's `pretrained_cfg`), with the overrides
`kwargs`; or `nothing` (the default) for [`create_processor`](@ref) to find the processor
files of the repository as transformers' `AutoProcessor` does.
"""
instantiate_processor(::ConfigFormat, files, config; kwargs...) = nothing

"""
    create_processor(name; revision = nothing, subfolder = "", kwargs...)

The processor of a checkpoint: it turns raw inputs (images, text, boxes) into the inputs of
the model, and the model's outputs back into results (boxes, masks, text), as the
checkpoint's configuration and its Python library do. [`create_model`](@ref) makes the
forward pass match PyTorch; the processor makes the whole path from raw input to result
match.

`name` is a Hub repository, a local directory or a bare architecture, as for
[`create_model`](@ref), and validated repositories default to the commit they were validated
at. The settings are read from the repository's files over the defaults of the Python class,
strictly: a setting that the port doesn't implement is an error, and an unknown key a
warning. The keyword arguments override them by name, e.g. `size = (; height = 512)`, where
a NamedTuple merges into a nested setting; each goes to the one component that has a
setting of that name, and a name that no component has is an error.

Returns the most complete processor the repository describes, as transformers'
`AutoProcessor` does: a processor that combines an image processor and a tokenizer, such as
`Transformers.Sam3Processor`; else an image processor, such as the
[`Timm.EvalTransform`](@ref Metalhead.Timm.EvalTransform) of every timm checkpoint; else a
tokenizer, `Transformers.TokenizersBackend`. Calling it returns what the model takes, a
NamedTuple (an array for `Timm.EvalTransform`):

```julia
proc = create_processor("unsloth/sam3.1")
inputs = proc(; images = img, text = "a cat")
outputs = model(inputs)
Transformers.post_process_instance_segmentation(proc, outputs;
    target_sizes = inputs.original_sizes)
```

Images are arrays of size `(width, height, channels)`, of integers from 0 to 255 or of reals
from 0 to 1, or, with ImageCore loaded, matrices of colorants (see
[`image_array`](@ref Metalhead.image_array)). Strings need a tokenizer backend
(`using HuggingFaceTokenizers`); token ids, as a tokenizer gives them (from 0), don't.
`show` prints each setting and where it comes from.
"""
function create_processor(name::AbstractString;
        revision::Union{Nothing, AbstractString} = nothing, subfolder::AbstractString = "",
        kwargs...)
    source = model_source(name, revision, String(subfolder))
    files = ProcessorFiles(source)
    config = read_json(files, "config.json")
    config === nothing && (config = read_json(files, "model_index.json"))
    found = config === nothing ? nothing : find_format(config)
    if found !== nothing
        p = instantiate_processor(first(found), files, config; kwargs...)
        p === nothing || return p
    end
    return load_processor(auto_processor_class(files, config), files; kwargs...)
end

function setting_repr(key, value)
    s = if key === :resample && value isa Integer && haskey(RESAMPLE_FILTERS, value)
        "$value ($(RESAMPLE_FILTERS[value]))"
    elseif value isa AbstractDict
        "$(length(value)) entries"
    else
        repr(value; context = :compact => true)
    end
    return length(s) > 72 ? first(s, 71) * "…" : s
end

# Where the setting `key` of `p`, of value `value`, comes from, as `show` prints it.
setting_origin(p, key, value) = origin_text(get(provenance(p), key, nothing), value)

function origin_text(o::Union{Nothing, Origin}, value)
    o === nothing && return ""
    isequal(o.value, value) || return "set after loading"
    o.source == "override" || return o.source
    r = o.replaced
    return "override ($(r.source): $(setting_repr(:value, r.value)))"
end

function Base.show(io::IO, p::Processor)
    print(io, nameof(typeof(p)), "(")
    join(io, ("$k = $(repr(v; context = io))" for (k, v) in pairs(components(p))), ", ")
    isempty(components(p)) || isempty(settings(p)) || print(io, "; ")
    join(io, ("$k = $(setting_repr(k, v))" for (k, v) in pairs(settings(p))), ", ")
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", p::Processor)
    return show_settings(io, p, "")
end

function show_settings(io, p, indent)
    print(io, nameof(typeof(p)))
    rows = [("$k = $(setting_repr(k, v))", setting_origin(p, k, v))
            for (k, v) in pairs(settings(p))]
    width = min(maximum(r -> length(first(r)), rows; init = 0), 52)
    for (setting, origin) in rows
        print(io, "\n", indent, "  ", rpad(setting, width), "  ", origin)
    end
    for (name, c) in pairs(components(p))
        print(io, "\n", indent, "  ", name, ": ")
        c isa Processor ? show_settings(io, c, indent * "  ") : show(io, c)
    end
end
