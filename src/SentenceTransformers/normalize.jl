# Ported from sentence_transformers.base.modules.normalize (sentence-transformers 6.2.0,
# Apache-2.0, UKP Lab and the Hugging Face Team).

"""
    Normalize(; module_input_name = "sentence_embedding",
              module_output_name = module_input_name)

sentence-transformers' `Normalize` module: the feature `module_input_name` of a NamedTuple
of features divided by its Euclidean norm along the first dimension (the features of each
text or token), as torch's `normalize(x, p = 2, dim = -1)`, as the feature
`module_output_name`.
"""
@concrete struct Normalize
    module_input_name
    module_output_name
end
Flux.@layer Normalize

function Normalize(; module_input_name = "sentence_embedding", module_output_name = nothing)
    return Normalize(String(module_input_name),
        String(something(module_output_name, module_input_name)))
end

function (m::Normalize)(features::NamedTuple)
    key = Symbol(m.module_input_name)
    haskey(features, key) || return features
    return merge(features,
        NamedTuple{(Symbol(m.module_output_name),)}((normalize(features[key]),)))
end

# torch's `normalize(x, p = 2, dim = -1)`, with its `eps = 1e-12`
function normalize(x)
    norm = sqrt.(sum(abs2, x; dims = 1))
    return x ./ max.(norm, eltype(x)(1e-12))
end

function load_normalize(source; pretrained, eltype)
    file = "config.json"
    c = read_settings(Normalize, read_file(source, file), file;
        defaults = (;
            module_input_name = "sentence_embedding", module_output_name = nothing))
    return Normalize(; c...)
end
MODULE_LOADERS["Normalize"] = load_normalize
