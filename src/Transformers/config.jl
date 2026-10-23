# The config.json of a transformers repository: its `model_type`, the class in `architectures`,
# and the fields of the model's config class, at the top level, among generic keys (token ids,
# generation settings, `transformers_version`, ...) that the models don't read.
struct TransformersFormat <: ConfigFormat end
const TRANSFORMERS = TransformersFormat()
Base.show(io::IO, ::TransformersFormat) = print(io, "transformers")

function architecture_of(::TransformersFormat, config)
    archs = get(config, "architectures", nothing)
    haskey(config, "model_type") && archs isa AbstractVector && !isempty(archs) ||
        return nothing
    length(archs) == 1 ||
        error("config.json names several architectures, $(join(archs, ", ")); expected one")
    return only(archs)
end

# The keys of config.json that the builder declares as keywords, as transformers ignores the others.
function model_args(::TransformersFormat, spec, config, kwargs)
    declared = Set(k for m in methods(spec.build) for k in Base.kwarg_decl(m))
    from_config = nt(filter(p -> Symbol(first(p)) in declared, config))
    return merge(spec.defaults, from_config, kwargs)
end
