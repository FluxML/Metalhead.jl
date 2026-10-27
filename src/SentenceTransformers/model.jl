# Ported from sentence_transformers.sentence_transformer.model and base.model
# (sentence-transformers 6.2.0, Apache-2.0, UKP Lab and the Hugging Face Team): a model's
# modules, prompts and `encode`, without training, evaluation, multi-process and quantized
# encoding, and similarity functions other than the model's own. Validated against
# test/fixtures/st_* and test/references/Qwen--Qwen3-Embedding-0.6B.

"""
    SentenceTransformer

A sentence-transformers model, as [`create_model`](@ref Metalhead.create_model) builds it
from a repository with a `modules.json`, such as `"Qwen/Qwen3-Embedding-0.6B"`: the
`modules` of its pipeline, applied in order to a NamedTuple of features, as
sentence-transformers' `SentenceTransformer`, with the settings of its
`config_sentence_transformers.json`.

  - The [`Transformer`](@ref Metalhead.SentenceTransformers.Transformer) module, `model[1]`,
    holds the ported transformers model (`model[1].auto_model`) and its tokenizer, and adds
    the `token_embeddings` of the texts.
  - [`Pooling`](@ref Metalhead.SentenceTransformers.Pooling) makes the `sentence_embedding`
    of each text from them, which
    [`Dense`](@ref Metalhead.SentenceTransformers.Dense) modules project and
    [`Normalize`](@ref Metalhead.SentenceTransformers.Normalize) normalizes.
  - `prompts`, the prompts of the model by name, such as the instruction of the queries of
    Qwen3-Embedding, and `default_prompt_name`, the name of the prompt of
    `encode` (`nothing` for none);
  - `similarity_fn_name`, the similarity of embeddings of
    [`similarity`](@ref Metalhead.SentenceTransformers.similarity): `"cosine"` (the
    default), `"dot"`, `"euclidean"` or `"manhattan"`;
  - `truncate_dim`, the number of features that `encode` keeps (`nothing` for all), for
    models trained to work with fewer (Matryoshka embeddings).

The keyword arguments of `create_model` override these settings, e.g.
`create_model(repo; truncate_dim = 256)`, and `prompts` adds to the model's.

`model(features)` applies the modules to the NamedTuple of `features` that
[`Metalhead.SentenceTransformers.preprocess`](@ref) makes of texts, with token indices
`input_ids` and their `attention_mask` (from 1, of size `(seq_len, batch)`), and returns
them with the outputs of each module: `token_embeddings` and `sentence_embedding`, of size
`(dim, batch)`. [`encode`](@ref Metalhead.SentenceTransformers.encode),
[`encode_query`](@ref Metalhead.SentenceTransformers.encode_query) and
[`encode_document`](@ref Metalhead.SentenceTransformers.encode_document) take the texts.

```julia
using Metalhead, HuggingFaceTokenizers
model = create_model("Qwen/Qwen3-Embedding-0.6B")
queries = encode_query(model, ["What is the capital of China?", "Explain gravity"])
documents = encode_document(model, ["The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other."])
similarity(model, queries, documents)   # 2 × 2, the largest on the diagonal
```
"""
@concrete struct SentenceTransformer
    modules
    prompts
    default_prompt_name
    similarity_fn_name
    truncate_dim
end
Flux.@layer :ignore SentenceTransformer trainable=(modules,)

Base.getindex(m::SentenceTransformer, i) = m.modules[i]
Base.length(m::SentenceTransformer) = length(m.modules)

function Base.show(io::IO, m::SentenceTransformer)
    names = join((nameof(typeof(x)) for x in m.modules), ", ")
    return print(io, "SentenceTransformer(", names, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::SentenceTransformer)
    print(io, "SentenceTransformer")
    for x in m.modules
        print(io, "\n  ", module_summary(x))
    end
    isempty(m.prompts) || print(io, "\n  prompts: ", join(keys(m.prompts), ", "))
    m.truncate_dim === nothing || print(io, "\n  truncate_dim = ", m.truncate_dim)
    return nothing
end

function module_summary(m::Transformer)
    model = nameof(typeof(m.auto_model))
    return "Transformer: $model, max_seq_length = $(m.max_seq_length)"
end
function module_summary(m::Pooling)
    prompt = m.include_prompt ? "" : ", without the prompt"
    return "Pooling: $(join(m.pooling_mode, ", "))$prompt"
end
function module_summary(m::Dense)
    out, in = size(m.linear.weight)
    residual = m.use_residual ? ", residual" : ""
    return "Dense: $in => $out, $(m.activation_function)$residual"
end
module_summary(::Normalize) = "Normalize"

const SIMILARITY_FUNCTIONS = ("cosine", "dot", "euclidean", "manhattan")

function sentence_transformer(;
        modules = nothing, prompts = (;), default_prompt_name = nothing,
        similarity_fn_name = nothing, truncate_dim = nothing)
    modules === nothing &&
        error("a SentenceTransformer is built from a repository with a modules.json, e.g. `create_model(\"Qwen/Qwen3-Embedding-0.6B\")`")
    first(modules) isa Transformer ||
        error("the first module of a SentenceTransformer is a Transformer, got a $(nameof(typeof(first(modules))))")
    prompts = (; (Symbol(k) => String(v) for (k, v) in pairs(something(prompts, (;))))...)
    default_prompt_name === nothing || haskey(prompts, Symbol(default_prompt_name)) ||
        throw(ArgumentError("the `default_prompt_name = $(repr(default_prompt_name))` is not a prompt of the model, whose prompts are $(join(repr.(String.(keys(prompts))), ", "))"))
    # sentence-transformers falls back on cosine for the similarities of other models
    # (multi-vector "maxsim")
    similarity_fn_name in SIMILARITY_FUNCTIONS || (similarity_fn_name = "cosine")
    truncate_dim === nothing || truncate_dim isa Integer && truncate_dim > 0 ||
        throw(ArgumentError("`truncate_dim` is a positive integer or `nothing`, got $(repr(truncate_dim))"))
    return SentenceTransformer(Tuple(modules), prompts,
        default_prompt_name === nothing ? nothing : String(default_prompt_name),
        similarity_fn_name, truncate_dim)
end

register!(ArchSpec(SENTENCE_TRANSFORMERS, "SentenceTransformer", sentence_transformer;
    family = "sentence_transformers",
    references = [
        "Qwen/Qwen3-Embedding-0.6B" => "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"]))

function (m::SentenceTransformer)(features::NamedTuple)
    check_inputs(m, features, (:input_ids,);
        hint = "make them with `Metalhead.SentenceTransformers.preprocess(model, texts)`, or encode texts with `encode(model, texts)`")
    for module_ in m.modules
        features = module_(features)
    end
    return features
end

"""
    preprocess(model::SentenceTransformer, texts; prompt = nothing)

The features of the vector of strings `texts` that `model(features)` takes, made by its
[`Transformer`](@ref Metalhead.SentenceTransformers.Transformer) module, with the string
`prompt` before each, as sentence-transformers' `preprocess`. Needs
`using HuggingFaceTokenizers`.
"""
function preprocess(m::SentenceTransformer, texts; kwargs...)
    return preprocess(first(m.modules), texts;
        kwargs...)
end

# The prompt of `encode`, as sentence-transformers' `_resolve_prompt`: `prompt`, else the
# prompt named `prompt_name`, else the default prompt.
function resolve_prompt(m::SentenceTransformer, prompt, prompt_name)
    prompt === nothing || return prompt
    if prompt_name !== nothing
        haskey(m.prompts, Symbol(prompt_name)) ||
            throw(ArgumentError("the model has no prompt named $(repr(prompt_name)); its prompts are $(join(repr.(String.(keys(m.prompts))), ", "))"))
        return m.prompts[Symbol(prompt_name)]
    end
    m.default_prompt_name === nothing && return nothing
    return get(m.prompts, Symbol(m.default_prompt_name), nothing)
end

"""
    encode(model::SentenceTransformer, texts; prompt_name = nothing, prompt = nothing,
           batch_size = 32, normalize_embeddings = false, truncate_dim = model.truncate_dim)

The embeddings of `texts`, a string or a vector of strings, as sentence-transformers'
`encode`: a matrix of size `(dim, length(texts))`, or a vector for one string, on the CPU.
`encode` is HuggingFaceTokenizers.jl's, which `using HuggingFaceTokenizers` brings in, and
which tokenizes them with the tokenizer of the model.

The texts are encoded with the string `prompt` before each, or with the prompt named
`prompt_name` of `model.prompts`, or with that of `model.default_prompt_name` if neither is
given; in batches of `batch_size` texts of similar lengths, on the device of the model. The
embeddings keep their first `truncate_dim` features, and are then normalized with
`normalize_embeddings = true`.
"""
function encode(m::SentenceTransformer, texts; prompt_name = nothing, prompt = nothing,
        batch_size::Integer = 32, normalize_embeddings::Bool = false,
        truncate_dim = m.truncate_dim)
    batch_size > 0 ||
        throw(ArgumentError("`batch_size` must be a positive integer, got $batch_size"))
    list, one_text = text_list(texts)
    prompt = resolve_prompt(m, prompt, prompt_name)
    device = Flux.get_device(first(m.modules).auto_model)
    # longest first, as sentence-transformers batches them, so that a batch pads little
    order = sortperm(length.(list); rev = true)
    embeddings = map(Iterators.partition(order, batch_size)) do batch
        features = device(preprocess(m, list[batch]; prompt))
        y = m(features).sentence_embedding
        truncate_dim === nothing || (y = y[1:min(truncate_dim, size(y, 1)), :])
        normalize_embeddings && (y = normalize(y))
        return Flux.cpu(y)
    end
    matrix = isempty(embeddings) ? zeros(Float32, 0, 0) : reduce(hcat, embeddings)
    matrix = matrix[:, invperm(order)]
    return one_text ? vec(matrix) : matrix
end

"""
    encode_query(model::SentenceTransformer, texts; kwargs...)

[`encode`](@ref Metalhead.SentenceTransformers.encode) for queries, as
sentence-transformers' `encode_query`: with the prompt named "query" if the model has one,
unless `prompt` or `prompt_name` is given.
"""
function encode_query(m::SentenceTransformer, texts; prompt_name = nothing,
        prompt = nothing, kwargs...)
    if prompt_name === nothing && prompt === nothing && haskey(m.prompts, :query)
        prompt_name = "query"
    end
    return encode(m, texts; prompt_name, prompt, kwargs...)
end

"""
    encode_document(model::SentenceTransformer, texts; kwargs...)

[`encode`](@ref Metalhead.SentenceTransformers.encode) for documents, as
sentence-transformers' `encode_document`: with the first prompt named "document", "passage"
or "corpus" that the model has, unless `prompt` or `prompt_name` is given.
"""
function encode_document(m::SentenceTransformer, texts; prompt_name = nothing,
        prompt = nothing, kwargs...)
    if prompt_name === nothing && prompt === nothing
        found = findfirst(k -> haskey(m.prompts, k), (:document, :passage, :corpus))
        found === nothing || (prompt_name = (:document, :passage, :corpus)[found])
    end
    return encode(m, texts; prompt_name, prompt, kwargs...)
end

"""
    similarity(model::SentenceTransformer, a, b)

The similarities of the embeddings `a` and `b` (matrices of size `(dim, n)`, or vectors) by
the similarity function of the model, `model.similarity_fn_name`, as sentence-transformers'
`similarity`: a matrix of size `(n_a, n_b)`, the cosine similarities, dot products, or
negative Euclidean or Manhattan distances of their columns.
"""
function similarity(m::SentenceTransformer, a::AbstractVecOrMat, b::AbstractVecOrMat)
    a, b = reshape(a, size(a, 1), :), reshape(b, size(b, 1), :)
    name = m.similarity_fn_name
    name == "cosine" && return normalize(a)' * normalize(b)
    name == "dot" && return a' * b
    a3, b3 = reshape(a, size(a, 1), :, 1), reshape(b, size(b, 1), 1, :)
    name == "euclidean" &&
        return -dropdims(sqrt.(sum(abs2, a3 .- b3; dims = 1)); dims = 1)
    name == "manhattan" ||
        throw(ArgumentError("`similarity_fn_name = $(repr(name))` is not supported; supported: $(join(repr.(SIMILARITY_FUNCTIONS), ", "))"))
    return -dropdims(sum(abs, a3 .- b3; dims = 1); dims = 1)
end
