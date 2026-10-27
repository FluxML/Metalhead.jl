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
    TextProcessor(tokenizer; model_input_names = ["input_ids", "attention_mask"])

The processor of a repository whose processor is its tokenizer, such as that of a language
model or a text encoder, where transformers' `AutoProcessor` returns the tokenizer:
HuggingFaceTokenizers.jl's `Tokenizer` of the repository's tokenizer.json, which
[`create_processor`](@ref Metalhead.create_processor) loads after `using
HuggingFaceTokenizers`, with the padding token of its tokenizer_config.json. Its one setting,
`model_input_names`, is that of transformers' tokenizer: the keys that it returns, in order,
from `"input_ids"`, `"token_type_ids"` and `"attention_mask"`. It is the
`model_input_names` of tokenizer_config.json, else that of the `tokenizer_class` there
(`BertTokenizer` returns `token_type_ids` too), else `["input_ids", "attention_mask"]`.

    (processor::TextProcessor)(text; text_pair = nothing, add_special_tokens = true)
    (processor::TextProcessor)(; text, text_pair = nothing, add_special_tokens = true)

The inputs of a language model or text encoder from `text`, a string or a vector of strings,
which the tokenizer encodes, with the special tokens that it adds around them unless
`add_special_tokens = false`, as a NamedTuple that the model takes as it is
(`model(inputs)`). `text_pair`, of the same form and size, encodes each text with a second
one, as transformers' `tokenizer(text, text_pair)` does with the template of the tokenizer
(`[CLS] a [SEP] b [SEP]` for BERT, `<s> a </s></s> b </s>` for XLM-RoBERTa): the input of a
cross-encoder, such as a reranker, for a query and a passage.

| Key              | Size               | Contents                                                                 |
|:---------------- |:------------------ |:------------------------------------------------------------------------ |
| `input_ids`      | `(seq_len, batch)` | token indices of the texts, 1-based, padded on the right                 |
| `token_type_ids` | `(seq_len, batch)` | 0 for the tokens of the text, 1 for those of its pair; 0 for the padding |
| `attention_mask` | `(seq_len, batch)` | 1 for the tokens, 0 for the padding                                      |

One text gives vectors `(seq_len,)`, so that the logits of its next token are
`model(inputs)[:, end]`. A batch of texts is padded on the right to the longest, with the
padding token of the tokenizer, as transformers pads it with `padding = true`; a tokenizer
without a padding token pads with the token index 1, which the `attention_mask` keeps out of
the attention, so that the outputs of each text are those it has alone. The texts are not
truncated, as transformers' tokenizers by default. `token_type_ids` are the segment ids of
transformers, 0 and 1, not token indices.

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

reranker = create_model("BAAI/bge-reranker-base")
proc = create_processor("BAAI/bge-reranker-base")
reranker(proc(["what is a panda?", "what is a panda?"];
    text_pair = ["The giant panda is a bear.", "Paris is in France."]))   # 1 × 2 scores
```
"""
@processor (tokenizer,) struct TextProcessor <: AbstractProcessor
    tokenizer::Any = nothing
    model_input_names::Any = ["input_ids", "attention_mask"]
    provenance::NamedTuple = (;)
end

const TEXT_INPUT_NAMES = ["input_ids", "token_type_ids", "attention_mask"]

function standardize_settings(::Type{TextProcessor}, settings::NamedTuple)
    names = settings.model_input_names
    names isa Union{AbstractVector, Tuple} && all(n -> n isa AbstractString, names) &&
    "input_ids" in names && issubset(names, TEXT_INPUT_NAMES) && allunique(names) ||
        throw(ArgumentError("`model_input_names` must be input names from $(TEXT_INPUT_NAMES), without repeats, with `\"input_ids\"`, got $(repr(names))"))
    return merge(settings, (; model_input_names = String[n for n in names]))
end

# The tokenizer classes of tokenizer_config.json whose tokenizers return `token_type_ids`
# (transformers' `model_input_names` of the class, which the file does not repeat)
const TOKEN_TYPE_TOKENIZERS = ("BertTokenizer", "BertTokenizerFast")

# The `model_input_names` of the tokenizer in `files`, as `(key, value, file)` entries of
# `processor_from_entries`: those of tokenizer_config.json, else of its `tokenizer_class`.
function tokenizer_input_names(files::ProcessorFiles)
    config = read_json(files, "tokenizer_config.json")
    config === nothing && return []
    names = get(config, "model_input_names", nothing)
    names === nothing ||
        return [("model_input_names", jsonvalue(names), "tokenizer_config.json")]
    class = get(config, "tokenizer_class", nothing)
    class in TOKEN_TYPE_TOKENIZERS || return []
    return [("model_input_names", TEXT_INPUT_NAMES,
        "tokenizer_config.json (`tokenizer_class`: $class)")]
end

function load_processor(::Type{TextProcessor}, files::ProcessorFiles; kwargs...)
    source = source_name(files.source)
    entries = tokenizer_input_names(files)
    processor_from_entries(TextProcessor, entries; kwargs...)  # the overrides, before the tokenizer
    tokenizers_extension() === nothing &&
        error("the processor of $source is a tokenizer, which needs HuggingFaceTokenizers.jl: run `using HuggingFaceTokenizers` before `create_processor`")
    tok = load_tokenizer(files)
    tok === nothing && error("$source has no tokenizer.json, the file of its tokenizer")
    return processor_from_entries(TextProcessor, entries; components = (; tokenizer = tok),
        kwargs...)
end

(p::TextProcessor)(; text, kwargs...) = p(text; kwargs...)

function (p::TextProcessor)(text; text_pair = nothing, add_special_tokens::Bool = true)
    texts, one_text = text_list(text)
    pairs = if text_pair === nothing
        fill(nothing, length(texts))
    else
        others, one_pair = text_list(text_pair)
        one_pair == one_text && length(others) == length(texts) ||
            throw(ArgumentError("`text_pair` must have as many texts as `text` ($(length(texts))), got $(length(others))"))
        others
    end
    tok = tokenizer_of(p)
    encodings = [encode_pair(tok, t, pair; add_special_tokens)
                 for (t, pair) in zip(texts, pairs)]
    padded = pad_ids([e.ids for e in encodings], something(pad_token_id(tok), 0))
    token_type_ids = zeros(Int, size(padded.input_ids))
    for (j, e) in enumerate(encodings)
        token_type_ids[eachindex(e.type_ids), j] .= e.type_ids
    end
    available = (; padded..., token_type_ids)
    inputs = NamedTuple{Tuple(Symbol.(p.model_input_names))}(
        map(k -> available[Symbol(k)], p.model_input_names))
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

# The ids and the segment ids (`type_ids`) of the tokens of `text`, or, if `pair` is not
# `nothing`, of `text` and `pair` as the tokenizer's template joins them, as a NamedTuple
# `(; ids, type_ids)`. A tokenizer that implements `encode_ids` only encodes single texts, with
# segment ids 0.
function encode_pair(tok, text, pair; add_special_tokens::Bool = true)
    pair === nothing ||
        throw(ArgumentError("$(nameof(typeof(tok))) can't encode a pair of texts: it has no method of `encode_pair`"))
    ids = encode_ids(tok, text; add_special_tokens)
    return (; ids, type_ids = zeros(Int, length(ids)))
end

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
