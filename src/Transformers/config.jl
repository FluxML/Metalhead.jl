# The config.json of a transformers repository: its `model_type`, the class in `architectures`,
# and the fields of the model's config class, at the top level, among generic keys (token ids,
# generation settings, `transformers_version`, ...) that the models don't read.
struct TransformersFormat <: AbstractConfigFormat end
const TRANSFORMERS = TransformersFormat()
direct_weight_init(::TransformersFormat) = true
Base.show(io::IO, ::TransformersFormat) = print(io, "transformers")

# The class that transformers' AutoModel builds for a `model_type` (`MODEL_MAPPING_NAMES`),
# of the ported classes: the architecture of a config.json without `architectures`, as
# those of the SigLIP 2 checkpoints of fixed resolution.
const MODEL_TYPE_CLASSES = Dict{String, String}()

function architecture_of(::TransformersFormat, config)
    haskey(config, "model_type") || return nothing
    archs = get(config, "architectures", nothing)
    archs isa AbstractVector && !isempty(archs) ||
        return get(MODEL_TYPE_CLASSES, config["model_type"], nothing)
    length(archs) == 1 ||
        error("config.json names several architectures, $(join(archs, ", ")); expected one")
    return only(archs)
end

# The keys of config.json that the builder declares as keywords, as transformers ignores the others.
function model_args(::TransformersFormat, spec, config, kwargs)
    return merge(spec.defaults, declared_args(spec.build, config), kwargs)
end
