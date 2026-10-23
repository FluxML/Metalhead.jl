"""
    ConfigFormat

How the `config.json` of a model repository names an architecture and its arguments: timm's
(`architecture`, `model_args`, `pretrained_cfg`), a future library's `architectures`, ... A
ported library defines a subtype and registers its architectures with it. Only
[`architecture_of`](@ref) is required; the other methods have defaults.
"""
abstract type ConfigFormat end

"""
    architecture_of(format, config) -> Union{Nothing, String}

The name of the architecture described by the parsed `config.json`, or `nothing` if the
configuration isn't in this `format`.
"""
architecture_of(::ConfigFormat, config) = nothing

"""
    model_args(format, spec, config, kwargs::NamedTuple) -> NamedTuple

The keyword arguments for `spec.build`: the defaults of the architecture, overridden by the
configuration and then by the user's `kwargs`.
"""
model_args(::ConfigFormat, spec, config, kwargs) = merge(spec.defaults, kwargs)

"""
    skip_prefix(format, config, args) -> Union{Nothing, String}

The prefix of the checkpoint keys that don't load into a model built with `args`, which keeps
its initial values (e.g. the classifier after changing `num_classes`). Throws if the
checkpoint can't be loaded into such a model at all.
"""
skip_prefix(::ConfigFormat, config, args) = nothing

"""
    ArchSpec(format, name, build; family = "other", defaults = (;),
             references = Pair{String,String}[], ignore = Regex[])

A ported architecture, as `config.json` files in `format` name it. `build(; kwargs...)`
constructs the model from the keyword arguments of the original library (e.g. timm's), merged
over `defaults`. `references` lists `repo => sha` checkpoints the port was validated against;
the first one is the default pretrained checkpoint. `ignore` lists extra checkpoint keys the
model doesn't use. `family` is the original library's family identifier, used to group
architectures in [`list_models`](@ref) and the documentation.
"""
struct ArchSpec
    format::ConfigFormat
    name::String
    family::String
    build::Any
    defaults::NamedTuple
    references::Vector{Pair{String, String}}
    ignore::Vector{Regex}
end

function ArchSpec(format::ConfigFormat, name::AbstractString, build;
        family::AbstractString = "other", defaults::NamedTuple = (;),
        references = Pair{String, String}[], ignore = Regex[])
    return ArchSpec(
        format, String(name), String(family), build, defaults,
        collect(Pair{String, String}, references),
        collect(Regex, ignore))
end

# Architecture names are unique across formats, so that `create_model("name")` is unambiguous.
const REGISTRY = Dict{String, ArchSpec}()
const FORMATS = ConfigFormat[]

function register!(spec::ArchSpec)
    old = get(REGISTRY, spec.name, nothing)
    old === nothing || old.format == spec.format ||
        error("architecture `$(spec.name)` is already registered for $(old.format)")
    spec.format in FORMATS || push!(FORMATS, spec.format)
    return REGISTRY[spec.name] = spec
end

function lookup(name::AbstractString)
    get(REGISTRY, String(name)) do
        return error("architecture `$name` has not been ported to Metalhead; see `list_models()`")
    end
end

# The format of a parsed config.json and the architecture it describes.
function identify(config)
    for format in FORMATS
        architecture = architecture_of(format, config)
        architecture === nothing || return format, architecture
    end
    return error("config.json is not in the format of a ported library " *
                 "($(join(FORMATS, ", ")))")
end

"""
    list_models(; pretrained = false, byfamily = false)

Names of the ported architectures, or with `pretrained = true` the Hub repositories
of the checkpoints they were validated against.

By default, returns a `Vector{String}`, ordered by architecture name. With `byfamily = true`,
returns a `Dict{String, Vector{String}}` mapping family identifiers to lists in the same
order.
For timm, the identifiers are its model module names, such as `"resnet"`,
`"vision_transformer"` and `"xception_aligned"`. With `pretrained = true`, families without
validated checkpoints are omitted.

```julia
families = list_models(; byfamily = true)
families["xception_aligned"]
list_models(; byfamily = true, pretrained = true)["resnet"]
```
"""
function list_models(; pretrained::Bool = false, byfamily::Bool = false)
    specs = sort!(collect(values(REGISTRY)); by = s -> s.name)
    if !byfamily
        pretrained || return [s.name for s in specs]
        return [first(r) for s in specs for r in s.references]
    end
    families = Dict{String, Vector{String}}()
    for spec in specs
        pretrained && isempty(spec.references) && continue
        names = get!(families, spec.family, String[])
        if pretrained
            append!(names, first(r) for r in spec.references)
        else
            push!(names, spec.name)
        end
    end
    return families
end
