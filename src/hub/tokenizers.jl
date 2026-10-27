# Metalhead ports no tokenizer. The tokenizer of a checkpoint is HuggingFaceTokenizers.jl's
# `Tokenizer` of the repository's tokenizer.json, which runs Hugging Face's tokenizers
# library; ext/MetalheadHuggingFaceTokenizersExt.jl loads it and implements the functions
# below, through which processors and pipelines use a tokenizer. A type with methods of
# `encode_ids`, `token_id` and `pad_token_id` (a test's) stands in for it. Token ids start
# at 0, as tokenizers give them; the models take token indices, which start at 1.

tokenizers_extension() = Base.get_extension(@__MODULE__, :MetalheadHuggingFaceTokenizersExt)

# The tokenizer of the processor files `files`: their tokenizer.json, at the commit of the
# other files, whose padding token is the `pad_token` of tokenizer_config.json unless
# tokenizer.json sets its padding. `nothing` without HuggingFaceTokenizers or without a
# tokenizer.json, as the component `tokenizer` of a processor is then.
function load_tokenizer(files::ProcessorFiles)
    tokenizers_extension() === nothing && return nothing
    return resolve_tokenizer(tokenizer_file(files))
end

# The tokenizer of `files` as `load_tokenizer` loads it, or, without HuggingFaceTokenizers,
# the `DeferredTokenizer` of its tokenizer.json, which loads once HuggingFaceTokenizers is
# loaded. A pipeline holds its tokenizer so.
function load_tokenizer_or_defer(files::ProcessorFiles)
    return tokenizers_extension() === nothing ? tokenizer_file(files) :
           load_tokenizer(files)
end

# A tokenizer.json and the padding token of its tokenizer, which `resolve_tokenizer` loads
# when HuggingFaceTokenizers is loaded.
struct DeferredTokenizer
    path::String
    pad::Union{Nothing, String}
end

function Base.show(io::IO, t::DeferredTokenizer)
    return print(io, "DeferredTokenizer(", repr(t.path), ")")
end

# The `DeferredTokenizer` of `files`, or `nothing` without a tokenizer.json.
function tokenizer_file(files::ProcessorFiles)
    config = read_json(files, "tokenizer_config.json")
    path = source_file_if_exists(files, "tokenizer.json")
    path === nothing && return nothing
    pad = config === nothing ? nothing : token_content(get(config, "pad_token", nothing))
    return DeferredTokenizer(path, pad)
end

# A tokenizer, or the tokenizer of a `DeferredTokenizer` if HuggingFaceTokenizers is loaded
# (else `nothing`).
resolve_tokenizer(tok) = tok
function resolve_tokenizer(t::DeferredTokenizer)
    ext = tokenizers_extension()
    return ext === nothing ? nothing : ext.load_tokenizer(t.path, t.pad)
end

"""
    TextProcessor(tokenizer)

The processor of a repository whose processor is its tokenizer, such as that of a language
model, where transformers' `AutoProcessor` returns the tokenizer: HuggingFaceTokenizers.jl's
`Tokenizer` of the repository's tokenizer.json, which [`create_processor`](@ref
Metalhead.create_processor) loads after `using HuggingFaceTokenizers`, with the padding
token of its tokenizer_config.json. It has no settings to override.

    (processor::TextProcessor)(text; add_special_tokens = true)
    (processor::TextProcessor)(; text, add_special_tokens = true)

The inputs of a language model from `text`, a string or a vector of strings, which the
tokenizer encodes, with the special tokens that it adds around them unless
`add_special_tokens = false`, as a NamedTuple that the model takes as it is
(`model(inputs)`):

| Key              | Size               | Contents                                                     |
|:---------------- |:------------------ |:------------------------------------------------------------ |
| `input_ids`      | `(seq_len, batch)` | token indices of the texts, 1-based, padded on the right     |
| `attention_mask` | `(seq_len, batch)` | 1 for the tokens, 0 for the padding                          |

One text gives vectors `(seq_len,)`, so that the logits of its next token are
`model(inputs)[:, end]`. A batch of texts is padded on the right to the longest, with the
padding token of the tokenizer, as transformers pads it with `padding = true`; a tokenizer
without a padding token pads with the token index 1, which the `attention_mask` keeps out of
the attention, so that the outputs of each text are those it has alone.

Token indices start at 1, the token ids of the tokenizer at 0. `decode(processor, indices)`
(HuggingFaceTokenizers' `decode`, which takes the keywords of the tokenizer's, such as
`skip_special_tokens`) decodes token indices, and `processor.tokenizer` is the tokenizer,
which takes and returns token ids.

```julia
using Metalhead, HuggingFaceTokenizers
model = create_model("HuggingFaceTB/SmolLM2-135M")
tok = create_processor("HuggingFaceTB/SmolLM2-135M")
logits = model(tok("The capital of France is"))
decode(tok, argmax(logits[:, end]))   # " Paris"
```
"""
@processor (tokenizer,) struct TextProcessor <: AbstractProcessor
    tokenizer::Any = nothing
    provenance::NamedTuple = (;)
end

function load_processor(::Type{TextProcessor}, files::ProcessorFiles; kwargs...)
    source = source_name(files.source)
    isempty(kwargs) ||
        throw(ArgumentError("the processor of $source is a tokenizer, which has no settings to override (got $(join(("`$k`" for k in keys(kwargs)), ", ")))"))
    tokenizers_extension() === nothing &&
        error("the processor of $source is a tokenizer, which needs HuggingFaceTokenizers.jl: run `using HuggingFaceTokenizers` before `create_processor`")
    tok = load_tokenizer(files)
    tok === nothing && error("$source has no tokenizer.json, the file of its tokenizer")
    return TextProcessor(tok)
end

(p::TextProcessor)(; text, add_special_tokens::Bool = true) = p(text; add_special_tokens)

function (p::TextProcessor)(text; add_special_tokens::Bool = true)
    texts, one_text = text_list(text)
    tok = tokenizer_of(p)
    ids = [encode_ids(tok, t; add_special_tokens) for t in texts]
    inputs = pad_ids(ids, something(pad_token_id(tok), 0))
    return one_text ? map(vec, inputs) : inputs
end

# The texts of the argument `text` of a processor, a string or a vector of strings, and
# whether it is one string.
function text_list(text)
    text isa AbstractString && return [text], true
    text isa AbstractVector && all(t -> t isa AbstractString, text) ||
        throw(ArgumentError("`text` is a string or a vector of strings, got a $(typeof(text))"))
    return collect(text), false
end

# The tokenizer of the processor `p`, which its text needs.
function tokenizer_of(p)
    p.tokenizer === nothing &&
        error("$(nameof(typeof(p))) has no tokenizer to encode text: run `using HuggingFaceTokenizers` before `create_processor`, from a repository with a tokenizer.json")
    return p.tokenizer
end

# A token of tokenizer_config.json: a string, or an `AddedToken` saved as a dictionary.
token_content(t::AbstractString) = String(t)
token_content(t::AbstractDict) = String(t["content"])
token_content(::Nothing) = nothing

# The ids of the tokens of `text`, with the special tokens that the tokenizer adds around
# them unless `add_special_tokens = false`.
function encode_ids end

# The id of the token `token`, or `nothing` if the vocabulary lacks it.
function token_id end

# The id of the padding token of `tok`, or `nothing` if it has none.
function pad_token_id end

# The token indices and the attention mask, of size `(len, batch)`, of a batch of texts,
# from the token ids `ids` of each: padded on the right with the id `pad_id`, to the longest
# text or to `max_length`, as transformers pads a batch. Longer texts are an error.
function pad_ids(ids::AbstractVector{<:AbstractVector{<:Integer}},
        pad_id::Union{Nothing, Integer}; max_length::Union{Nothing, Integer} = nothing)
    n = something(max_length, maximum(length, ids; init = 0))
    input_ids = zeros(Int, n, length(ids))
    attention_mask = zeros(Int, n, length(ids))
    for (j, x) in enumerate(ids)
        m = length(x)
        m <= n || throw(ArgumentError("a text has $m tokens, more than the $n to pad to"))
        if m < n
            pad_id === nothing &&
                throw(ArgumentError("the tokenizer has no padding token, to pad a text of $m tokens to $n"))
            input_ids[(m + 1):n, j] .= pad_id + 1
        end
        input_ids[1:m, j] .= x .+ 1
        attention_mask[1:m, j] .= 1
    end
    return (; input_ids, attention_mask)
end
