"""
    ArchSpec(library, name, build; defaults = (;), references = Pair{String,String}[], ignore = Regex[])

A ported architecture. `build(; kwargs...)` constructs the model from the keyword arguments of
the original library (e.g. timm's), merged over `defaults`. `references` lists `repo => sha`
checkpoints the port was validated against; the first one is the default pretrained checkpoint.
`ignore` lists extra checkpoint keys the model doesn't use.
"""
struct ArchSpec
    library::Symbol
    name::String
    build::Any
    defaults::NamedTuple
    references::Vector{Pair{String, String}}
    ignore::Vector{Regex}
end

function ArchSpec(library::Symbol, name::AbstractString, build; defaults::NamedTuple = (;),
        references = Pair{String, String}[], ignore = Regex[])
    return ArchSpec(
        library, String(name), build, defaults, collect(Pair{String, String}, references),
        collect(Regex, ignore))
end

const REGISTRY = Dict{Tuple{Symbol, String}, ArchSpec}()

register!(spec::ArchSpec) = (REGISTRY[(spec.library, spec.name)] = spec)

function lookup(library::Symbol, name::AbstractString)
    get(REGISTRY, (library, String(name))) do
        return error("architecture `$name` from $library has not been ported to Metalhead; see `list_models()`")
    end
end

"""
    list_models(; pretrained = false, library = :timm) -> Vector{String}

Names of the ported architectures, or with `pretrained = true` the Hub repositories
of the checkpoints they were validated against.
"""
function list_models(; pretrained::Bool = false, library::Symbol = :timm)
    specs = sort!([s for s in values(REGISTRY) if s.library == library]; by = s -> s.name)
    pretrained || return [s.name for s in specs]
    return [first(r) for s in specs for r in s.references]
end
