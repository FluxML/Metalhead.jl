# Ported from transformers.tokenization_utils_tokenizers and tokenization_utils_base
# (transformers 5.20, Apache-2.0, The Hugging Face Team): `TokenizersBackend`, the tokenizer
# of transformers 5 for a tokenizer.json, with the settings of tokenizer_config.json
# (special tokens, padding, truncation) read as `from_pretrained` reads them, over the
# defaults of the class that `AutoTokenizer` builds. Tokenizing a string, a tokenization
# algorithm, is left to a backend: the HuggingFaceTokenizers.jl extension, or any type with
# methods of `encode_ids` and `decode_ids`. Validated against
# test/references/processors/sam3.safetensors and tokenizers.safetensors.

include("tokenization_auto.jl")

"""
    encode_ids(backend, text::AbstractString; add_special_tokens = true) -> Vector{Int}

The token ids (from 0, as Hugging Face's tokenizers number them) of `text`, with the special
tokens of the tokenizer's post-processor if `add_special_tokens`. A tokenizer backend
defines it, as the extension of Metalhead for HuggingFaceTokenizers.jl does for its
`Tokenizer`; pass one to [`create_processor`](@ref Metalhead.create_processor) as
`tokenizer = backend`.
"""
function encode_ids end

"""
    decode_ids(backend, ids::AbstractVector{<:Integer}; skip_special_tokens = false)
        -> String

The text of the token ids `ids` (from 0), without the special tokens if
`skip_special_tokens`: the method of a tokenizer backend for [`encode_ids`](@ref
Metalhead.Transformers.encode_ids).
"""
function decode_ids end

"""
    TokenizersBackend(; tokenizer = nothing, tokenizer_file = nothing, pad_token = nothing,
                      padding_side = "right", ...)

transformers' tokenizer of a tokenizer.json, which [`create_processor`](@ref
Metalhead.create_processor) loads with the settings of the repository's
tokenizer_config.json: the processor of a language model such as `"Qwen/Qwen3-0.6B"`, and
the `tokenizer` of a processor or of a diffusers pipeline. Its fields are transformers'
attributes, with the defaults of `tokenizer_class`, the class that transformers'
`AutoTokenizer` builds (`GPT2Tokenizer`, `Qwen2Tokenizer`, ...; loading a class that
Metalhead hasn't validated warns).

`tokenizer` is the backend that encodes and decodes strings: `nothing` until a string needs
it, when the HuggingFaceTokenizers.jl extension loads `tokenizer_file`, or yours (see
[`encode_ids`](@ref Metalhead.Transformers.encode_ids)). `add_bos_token` and
`add_eos_token` are `nothing` for the special tokens of tokenizer.json, or set at creation
(`create_processor(repo; add_bos_token = false)`) for those they ask for. `init_kwargs` are
the keywords of calls that tokenizer_config.json sets, which processors that combine the
tokenizer (`Qwen3VLProcessor`) take as the defaults of their calls, as transformers does.

    (tok::TokenizersBackend)(text; padding = nothing, truncation = nothing,
                             max_length = nothing, padding_side = nothing,
                             add_special_tokens = true)

The inputs of a language model for `text`, a string, the token ids of a tokenizer (from 0,
with its special tokens, which need no backend), or a vector of them:

| Key              | Size                                          | Contents                                                                 |
|:---------------- |:--------------------------------------------- |:------------------------------------------------------------------------ |
| `input_ids`      | `(seq_len, batch)`, `(seq_len,)` for one text | token indices, 1-based, padded on the side `padding_side`                |
| `attention_mask` | as `input_ids`                                | 1 for the tokens, 0 for the padding; unless `model_input_names` omits it |

The keywords are transformers': `padding` is `true` (`:longest`), `:max_length` or `false`,
and needs a `pad_token` (`create_processor(repo; pad_token = ...)`); `truncation` cuts the
tokens of a string, keeping its special tokens; `max_length` defaults to
`model_max_length`. Without `padding`, a batch of different lengths is padded to the
longest, where transformers fails. [`decode`](@ref Metalhead.Transformers.decode) turns
indices back into text.

```julia
using Metalhead, HuggingFaceTokenizers
using Metalhead.Transformers: decode
tok = create_processor("unsloth/Llama-3.2-1B")
inputs = tok(["The capital of France is", "Hello"])   # padded on the left, as Llama 3's
model = create_model("unsloth/Llama-3.2-1B"; pretrained = true)
logits = model(inputs)                                # (vocab_size, 6, 2)
[decode(tok, [argmax(l)]) for l in eachcol(logits[:, end, :])]   # the next tokens
```
"""
Metalhead.@processor struct TokenizersBackend <: Processor
    tokenizer::Any = nothing
    tokenizer_file::Any = nothing
    tokenizer_class::String = "TokenizersBackend"
    bos_token::Union{Nothing, String} = nothing
    eos_token::Union{Nothing, String} = nothing
    unk_token::Union{Nothing, String} = nothing
    sep_token::Union{Nothing, String} = nothing
    pad_token::Union{Nothing, String} = nothing
    cls_token::Union{Nothing, String} = nothing
    mask_token::Union{Nothing, String} = nothing
    extra_special_tokens::Vector{String} = String[]
    model_specific_special_tokens::NamedTuple = (;)
    added_tokens_decoder::Dict{Int, String} = Dict{Int, String}()
    model_max_length::Union{Nothing, Int} = nothing
    padding_side::String = "right"
    truncation_side::String = "right"
    clean_up_tokenization_spaces::Bool = false
    add_bos_token::Union{Nothing, Bool} = nothing
    add_eos_token::Union{Nothing, Bool} = nothing
    add_prefix_space::Bool = false
    model_input_names::Vector{String} = ["input_ids", "attention_mask"]
    init_kwargs::NamedTuple = (;)
    provenance::NamedTuple = (;)
end

register_processor!("TokenizersBackend", TokenizersBackend)

# `gpu`, `f16` and the other Functors traversals of a model leave its tokenizer (that of a
# pipeline) as it is, with a backend that may be a Python object.
Functors.@leaf TokenizersBackend

const SPECIAL_TOKENS = (:bos_token, :eos_token, :unk_token, :sep_token, :pad_token,
    :cls_token, :mask_token)

# The keywords of transformers' processor calls (`TextKwargs`) that tokenizer_config.json
# can set: `ProcessorMixin._merge_kwargs` takes them as defaults of the calls
# (`init_kwargs`). `add_special_tokens` is one, but transformers fails on it, the name of a
# method.
const TEXT_KWARGS = (:padding, :truncation, :max_length, :stride,
    :is_split_into_words, :pad_to_multiple_of, :return_token_type_ids,
    :return_attention_mask, :return_overflowing_tokens, :return_special_tokens_mask,
    :return_offsets_mapping, :return_length, :return_mm_token_type_ids,
    :return_text_replacement_offsets, :return_tensors)

const CHAT_KEY = "chat templates are out of scope"
const BERT_KEY = "configures the normaliser of BERT's tokenizer classes, which Metalhead " *
                 "hasn't validated"
const SENTENCEPIECE_KEY = "configures sentencepiece tokenizer classes (Llama 2, T5), " *
                          "which Metalhead hasn't validated"
const CALL_KEY = "a keyword of its call: pass it to the call, or to the calls of a " *
                 "processor as the tokenizer's `init_kwargs`"

function ignored_keys(::Type{TokenizersBackend})
    return (; chat_template = CHAT_KEY, use_default_system_prompt = CHAT_KEY,
        response_template = CHAT_KEY, response_schema = CHAT_KEY,
        chat_control_tokens = CHAT_KEY, do_lower_case = BERT_KEY, strip_accents = BERT_KEY,
        tokenize_chinese_chars = BERT_KEY, do_basic_tokenize = BERT_KEY,
        never_split = BERT_KEY, legacy = SENTENCEPIECE_KEY,
        sp_model_kwargs = SENTENCEPIECE_KEY,
        errors = "stored but not read by transformers 5, whose decoder of bytes replaces " *
                 "the invalid ones",
        pretokenize_regex = "not read by Qwen3_5Tokenizer, which pre-tokenizes with its " *
                            "own expression, that of tokenizer.json",
        vocab_size = "the size of the vocabulary of tokenizer.json",
        spaces_between_special_tokens = "used by transformers' slow tokenizers only",
        truncation_strategy = "not read by transformers 5",
        pad_token_type_id = "the type id of padding tokens, which Metalhead doesn't return",
        name_or_path = "the name of the repository",
        tokenizer_file = "a file of the repository",
        vocab_file = "a file of the repository", merges_file = "a file of the repository",
        verbose = "logging",
        backend = "the library of the tokenizer", files_loaded = "the files loaded",
        (k => CALL_KEY for k in TEXT_KWARGS)...)
end
restricted_keys(::Type{TokenizersBackend}) = (; split_special_tokens = (false,),
    clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = (false,))
legacy_keys(::Type{TokenizersBackend}) = (; max_len = (:model_max_length,))

function check_processor(tok::TokenizersBackend)
    for side in (:padding_side, :truncation_side)
        getfield(tok, side) in ("right", "left") ||
            error("`$side = $(repr(getfield(tok, side)))` must be \"right\" or \"left\"")
    end
    issubset(tok.model_input_names, ("input_ids", "attention_mask")) ||
        error("`model_input_names = $(repr(tok.model_input_names))` is not supported (supported: input_ids, attention_mask)")
    for k in keys(tok.init_kwargs)
        k in TEXT_KWARGS ||
            error("`init_kwargs` has `$k`, which isn't a keyword of transformers' processors (`TextKwargs`: $(join(TEXT_KWARGS, ", ")))")
    end
    return nothing
end

# The defaults that the `__init__` of transformers' tokenizer classes and their class
# attributes set, for the classes that Metalhead validates: those of the validated
# repositories, whose tokenizer.json tokenizes strings as the class, which rebuilds its
# pipeline, does. LlamaTokenizer isn't one: it pre-tokenizes Llama 2's tokenizer.json
# otherwise.
const TOKENIZER_CLASSES = Dict(
    "TokenizersBackend" => (;),
    "GPT2Tokenizer" => (; unk_token = "<|endoftext|>", bos_token = "<|endoftext|>",
        eos_token = "<|endoftext|>"),
    "Qwen2Tokenizer" => (; unk_token = "<|endoftext|>", eos_token = "<|endoftext|>",
        pad_token = "<|endoftext|>"),
    "Qwen3_5Tokenizer" => (; unk_token = "<|endoftext|>", eos_token = "<|endoftext|>",
        pad_token = "<|endoftext|>"),
    "GemmaTokenizer" => (; unk_token = "<unk>", bos_token = "<bos>", eos_token = "<eos>",
        pad_token = "<pad>", mask_token = "<mask>", padding_side = "left"),
    "CLIPTokenizer" => (; unk_token = "<|endoftext|>", bos_token = "<|startoftext|>",
        eos_token = "<|endoftext|>", pad_token = "<|endoftext|>"))

# Classes that rebuild the pre-tokenizer of tokenizer.json with their `add_prefix_space`.
const PREFIX_SPACE_CLASSES = ("GPT2Tokenizer", "Qwen2Tokenizer", "Qwen3_5Tokenizer")

const WARNED_CLASSES = Set{String}()
const WARNED_CLASSES_LOCK = ReentrantLock()

function warn_unvalidated(class, source)
    first_time = lock(WARNED_CLASSES_LOCK) do
        class in WARNED_CLASSES && return false
        push!(WARNED_CLASSES, class)
        return true
    end
    validated = join(sort!(collect(keys(TOKENIZER_CLASSES))), ", ")
    first_time &&
        @warn "$class, the tokenizer class of $source, hasn't been validated in Metalhead: it has the defaults of TokenizersBackend, and tokenizer.json tokenizes its strings, which $class may do otherwise (validated: $validated)"
    return nothing
end

# A special token of a configuration: its content, or the `content` of an AddedToken.
token_content(t::AbstractString) = String(t)
token_content(t::AbstractDict) = String(t["content"])
token_content(::Nothing) = nothing

falsy(x) = x === nothing || x isa Union{AbstractVector, AbstractDict} && isempty(x)

# A named special token of a model (`image_token`), which isn't one of SPECIAL_TOKENS.
is_named_token(k, v) = endswith(k, "_token") && !(Symbol(k) in SPECIAL_TOKENS) &&
                       v isa Union{AbstractString, AbstractDict}

# The settings of a tokenizer as transformers' `_from_pretrained` and `__init__` read them,
# as entries of `processor_from_entries`: tokenizer_config.json, whose
# `additional_special_tokens` stand for empty `extra_special_tokens`, and only without its
# `added_tokens_decoder` the legacy special_tokens_map.json, which overrides it (and adds to
# a list of extra special tokens), and added_tokens.json. A `null` special token replaces
# the default of the class.
function tokenizer_entries(files::ProcessorFiles)
    kw = Dict{String, Tuple{Any, String}}()
    order = String[]
    set!(k, v, file) = (haskey(kw, k) || push!(order, k); kw[k] = (v, file))
    value(k) = first(get(kw, k, (nothing, "")))
    config = read_json(files, "tokenizer_config.json")
    config === nothing || foreach(((k, v),) -> set!(k, v, "tokenizer_config.json"), config)
    if haskey(kw, "additional_special_tokens") && falsy(value("extra_special_tokens"))
        set!("extra_special_tokens", kw["additional_special_tokens"]...)
        delete!(kw, "additional_special_tokens")
    end
    added = Dict{Int, String}()
    if haskey(kw, "added_tokens_decoder")
        for (id, token) in value("added_tokens_decoder")
            added[parse(Int, id)] = token_content(token)
        end
    else
        for (k, v) in something(read_json(files, "special_tokens_map.json"), ())
            old = value(k)
            k == "extra_special_tokens" && v isa AbstractVector &&
                old isa AbstractVector && (v = unique!([old; v]))
            set!(k, v, "special_tokens_map.json")
        end
        for (token, id) in something(read_json(files, "added_tokens.json"), ())
            added[id] = token
        end
    end
    # `extra_special_tokens` as a dictionary names the special tokens of a model, which
    # `_from_pretrained` moves to `model_specific_special_tokens`; `__init__` then takes
    # `additional_special_tokens` as the extra special tokens (Qwen3.5)
    named = Pair{Symbol, String}[]
    extra = value("extra_special_tokens")
    if extra isa AbstractDict
        append!(named, Symbol(n) => token_content(t) for (n, t) in extra)
        delete!(kw, "extra_special_tokens")
    end
    if haskey(kw, "additional_special_tokens") && !haskey(kw, "extra_special_tokens")
        set!("extra_special_tokens", kw["additional_special_tokens"]...)
    end
    entries = Tuple{String, Any, String}[]
    init_kwargs = Pair{Symbol, Any}[]
    for k in unique(order)
        haskey(kw, k) || continue
        v, file = kw[k]
        # transformers drops `add_bos_token` and `add_eos_token` of the files when there
        # is a tokenizer.json, whose post-processor adds the special tokens
        if k in ("additional_special_tokens", "added_tokens_decoder", "add_bos_token",
            "add_eos_token")
            continue
        elseif k == "extra_special_tokens"
            falsy(v) || push!(entries, (k, token_content.(v), file))
        elseif Symbol(k) in SPECIAL_TOKENS
            push!(entries, (k, v === nothing ? Some(nothing) : token_content(v), file))
        elseif Symbol(k) in TEXT_KWARGS
            v === nothing || push!(init_kwargs, Symbol(k) => jsonvalue(v))
        elseif is_named_token(k, v)
            push!(named, Symbol(k) => token_content(v))
        elseif k == "model_max_length" || k == "max_len"
            # transformers' "no limit" is 1e30, above 1e20
            push!(entries, (k, v === nothing || v > 1.0e20 ? nothing : Int(v), file))
        elseif k == "model_input_names"
            push!(entries, (k, v === nothing ? nothing : String.(v), file))
        else
            push!(entries, (k, jsonvalue(v), file))
        end
    end
    isempty(named) || push!(entries, ("model_specific_special_tokens",
        (; sort(named; by = first)...), "tokenizer_config.json"))
    isempty(added) ||
        push!(entries, ("added_tokens_decoder", added, "tokenizer_config.json"))
    isempty(init_kwargs) ||
        push!(entries, ("init_kwargs", (; init_kwargs...), "tokenizer_config.json"))
    return entries
end

function load_processor(::Type{TokenizersBackend}, files::ProcessorFiles; kwargs...)
    has_file(files, "tokenizer_config.json") || has_file(files, "tokenizer.json") ||
        error("$(Metalhead.source_name(files.source)) has no tokenizer (tokenizer_config.json or tokenizer.json)")
    config = something(read_json(files, "config.json"), Dict{String, Any}())
    tokenizer_config = something(read_json(files, "tokenizer_config.json"),
        Dict{String, Any}())
    class, decided_by = tokenizer_class(get(tokenizer_config, "tokenizer_class", nothing),
        get(config, "tokenizer_class", nothing), get(config, "model_type", nothing))
    defaults = get(TOKENIZER_CLASSES, class, nothing)
    defaults === nothing && warn_unvalidated(class, Metalhead.source_name(files.source))
    entries = Tuple{String, Any, String}[(String(k), v, "class default ($class)")
                                         for (k, v) in pairs(something(defaults, (;)))]
    append!(entries, filter(e -> e[1] != "tokenizer_class", tokenizer_entries(files)))
    decided_by === nothing || push!(entries, ("tokenizer_class", class, decided_by))
    push!(entries,
        ("tokenizer_file", SourceFile(files.source, "tokenizer.json"), "repository"))
    return processor_from_entries(TokenizersBackend, entries; kwargs...)
end

function Metalhead.setting_origin(tok::TokenizersBackend, key, value)
    key === :tokenizer && value === nothing && tok.tokenizer_file !== nothing &&
        return "loaded from tokenizer_file when a string needs it"
    return Metalhead.origin_text(get(tok.provenance, key, nothing), value)
end

# What Metalhead reads of a tokenizer.json, once per file: the type of its model, whether it
# pads or truncates, the `add_prefix_space` of its pre-tokenizer, and its added tokens.
struct TokenizerJSON
    model_type::String
    pads::Bool
    truncates::Bool
    add_prefix_space::Union{Nothing, Bool}
    added_tokens::Dict{String, Int}
end

const TOKENIZER_JSONS = Dict{String, TokenizerJSON}()
const VOCABULARY_IDS = Dict{Tuple{String, String}, Union{Nothing, Int}}()
const TOKENIZER_JSONS_LOCK = ReentrantLock()

function tokenizer_json(path)
    return lock(TOKENIZER_JSONS_LOCK) do
        get!(TOKENIZER_JSONS, path) do
            json = JSON.parsefile(path)
            added = Dict{String, Int}()
            for t in get(json, "added_tokens", ())
                get!(added, t["content"], t["id"])
            end
            return TokenizerJSON(model_type(json["model"]),
                get(json, "padding", nothing) !== nothing,
                get(json, "truncation", nothing) !== nothing,
                prefix_space(get(json, "pre_tokenizer", nothing)), added)
        end
    end
end

# The type of a model of tokenizer.json, which older files leave to its keys.
function model_type(model)
    haskey(model, "type") && return String(model["type"])
    haskey(model, "merges") && return "BPE"
    get(model, "vocab", nothing) isa AbstractVector && return "Unigram"
    return haskey(model, "max_input_chars_per_word") ? "WordPiece" : "WordLevel"
end

prefix_space(::Nothing) = nothing
function prefix_space(pre_tokenizer)
    type = get(pre_tokenizer, "type", nothing)
    type == "ByteLevel" && return pre_tokenizer["add_prefix_space"]
    type == "Sequence" || return nothing
    found = filter(!isnothing, map(prefix_space, pre_tokenizer["pretokenizers"]))
    return isempty(found) ? nothing : first(found)
end

# The id of `token` in the vocabulary of tokenizer.json, or `nothing`.
function vocabulary_id(path, token)
    return lock(TOKENIZER_JSONS_LOCK) do
        get!(VOCABULARY_IDS, (path, token)) do
            vocab = get(JSON.parsefile(path)["model"], "vocab", ())
            vocab isa AbstractDict && return get(vocab, token, nothing)
            i = findfirst(v -> (v isa AbstractVector ? first(v) : v) == token, vocab)
            return i === nothing ? nothing : i - 1
        end
    end
end

# transformers' tokenizer classes read these of tokenizer.json in ways that Metalhead
# doesn't port: padding and truncation are the defaults of every call, and the classes that
# rebuild the pre-tokenizer of tokenizer.json do it with their own `add_prefix_space`.
function check_tokenizer_json(tok::TokenizersBackend, path)
    json = tokenizer_json(path)
    for (sets, what) in ((json.pads, "padding"), (json.truncates, "truncation"))
        sets &&
            error("the tokenizer.json of $(tok.tokenizer_class) ($path) sets `$what`, which transformers applies to every call of the tokenizer: Metalhead doesn't support such a tokenizer.json")
    end
    tok.tokenizer_class in PREFIX_SPACE_CLASSES || return nothing
    json.add_prefix_space === nothing || json.add_prefix_space == tok.add_prefix_space ||
        error("`add_prefix_space = $(tok.add_prefix_space)` isn't supported: $(tok.tokenizer_class) rebuilds the pre-tokenizer of tokenizer.json with it, and that of $path has `add_prefix_space = $(json.add_prefix_space)`, which Metalhead tokenizes with")
    return nothing
end

# The backend of a tokenizer: its own, or the HuggingFaceTokenizers.jl tokenizer of its
# tokenizer.json, which the extension loads once.
function backend(tok::TokenizersBackend)
    tok.tokenizer === nothing || return tok.tokenizer
    tok.tokenizer_file === nothing &&
        error("$(tok.tokenizer_class) has no tokenizer.json to encode or decode strings: pass token ids, or a backend as `tokenizer`")
    ext = Base.get_extension(Metalhead, :MetalheadHuggingFaceTokenizersExt)
    ext === nothing &&
        error("encoding or decoding strings needs a tokenizer backend: run `using HuggingFaceTokenizers`, or pass token ids")
    path = Metalhead.local_path(tok.tokenizer_file)
    check_tokenizer_json(tok, path)
    return Base.invokelatest(ext.load_tokenizer, path)
end

# The id (from 0) of a token, as transformers' `convert_tokens_to_ids` finds it, or
# `nothing`: an added token of the configuration, else a single token of the backend `b`
# (that of the tokenizer, if it holds one), else an added token or a token of the
# vocabulary of tokenizer.json, which needs no backend.
function find_token_id(tok::TokenizersBackend, token, b = tok.tokenizer)
    ids = [id for (id, t) in tok.added_tokens_decoder if t == token]
    isempty(ids) || return minimum(ids)
    if b !== nothing
        ids = encode_ids(b, token; add_special_tokens = false)
        return length(ids) == 1 ? Int(only(ids)) : nothing
    end
    path = tok.tokenizer_file === nothing ? nothing :
           Metalhead.local_path_if_exists(tok.tokenizer_file)
    path === nothing && return nothing
    id = get(tokenizer_json(path).added_tokens, token, nothing)
    return id === nothing ? vocabulary_id(path, token) : id
end

function token_id(tok::TokenizersBackend, token, b = tok.tokenizer)
    id = find_token_id(tok, token, b)
    id === nothing || return id
    known = b !== nothing ? "its backend" :
            tok.tokenizer_file === nothing ?
            "a tokenizer.json or a backend, which it lacks" :
            "its tokenizer.json ($(tok.tokenizer_file))"
    return error("`$token` is not a token of $(tok.tokenizer_class): neither an added token (`added_tokens_decoder`) nor a token of $known")
end

function check_pad_token(tok::TokenizersBackend)
    tok.pad_token === nothing || return nothing
    eos = tok.eos_token === nothing ? "" : ", e.g. `pad_token = $(repr(tok.eos_token))`"
    return error("$(tok.tokenizer_class) has no padding token (`pad_token`) to pad sequences: set one when creating the processor$eos, as transformers asks")
end

pad_token_id(tok::TokenizersBackend) = (check_pad_token(tok); token_id(tok, tok.pad_token))

# Tokenizer classes whose `__init__` sets the post-processor after transformers'
# `update_post_processor`, so that `add_bos_token` and `add_eos_token` change nothing.
const FIXED_POST_PROCESSOR = ("CLIPTokenizer",)

# The ids of a string with its special tokens: those that the post-processor of
# tokenizer.json adds or, if `add_bos_token` or `add_eos_token` is set, the BOS and EOS
# tokens that they ask for around the tokens of the text, as transformers'
# `update_post_processor` replaces the post-processor (an unset one counts as `false`).
function encode_text(tok::TokenizersBackend, b, text, add_special_tokens)
    template = add_special_tokens &&
               (tok.add_bos_token !== nothing || tok.add_eos_token !== nothing) &&
               !(tok.tokenizer_class in FIXED_POST_PROCESSOR)
    template || return collect(Int, encode_ids(b, text; add_special_tokens))
    ids = collect(Int, encode_ids(b, text; add_special_tokens = false))
    something(tok.add_bos_token, false) && tok.bos_token !== nothing &&
        pushfirst!(ids, token_id(tok, tok.bos_token, b))
    something(tok.add_eos_token, false) && tok.eos_token !== nothing &&
        push!(ids, token_id(tok, tok.eos_token, b))
    return ids
end

# The ids of `text`, truncated to `max_length` (unless `nothing`) as the tokenizers library
# truncates: the tokens of the text, without those that the post-processor adds.
function token_ids(tok::TokenizersBackend, text::AbstractString, add_special_tokens,
        max_length)
    b = backend(tok)
    ids = encode_text(tok, b, text, add_special_tokens)
    (max_length === nothing || length(ids) <= max_length) && return ids
    content = add_special_tokens ?
              collect(Int, encode_ids(b, text; add_special_tokens = false)) : ids
    n = length(content)
    k = findfirst(k -> view(ids, (k + 1):(k + n)) == content, 0:(length(ids) - n))
    k === nothing &&
        error("the special tokens of $(tok.tokenizer_class) change the tokens of a text, which can't be truncated")
    k -= 1
    keep = max_length - (length(ids) - n)
    keep >= 0 ||
        error("`max_length = $max_length` is less than the $(length(ids) - n) special tokens of a sequence")
    kept = tok.truncation_side == "right" ? content[1:keep] : content[(end - keep + 1):end]
    return [ids[1:k]; kept; ids[(k + n + 1):end]]
end

function token_ids(tok::TokenizersBackend, ids::AbstractVector{<:Integer}, _, max_length)
    max_length === nothing || length(ids) <= max_length ||
        error("token ids of $(length(ids)) tokens are longer than `max_length = $max_length`; truncate them, or pass the text")
    return collect(Int, ids)
end

is_one_text(x) = x isa AbstractString || x isa AbstractVector{<:Integer}

# transformers' `_get_padding_truncation_strategies`, for single sequences.
# `padding = nothing` is transformers' default, `false`, except that a batch of different
# lengths is padded to the longest.
function call_strategies(tok, padding, truncation, max_length)
    max_length !== nothing && padding in (nothing, false) && truncation === nothing &&
        (truncation = true)
    pad = padding === nothing || padding === true ? :longest :
          padding === false ? :do_not_pad : Symbol(padding)
    pad in (:longest, :max_length, :do_not_pad) ||
        error("`padding = $(repr(padding))` must be true, false, :longest, :max_length or :do_not_pad")
    trunc = truncation === nothing || truncation === false ? :do_not_truncate :
            truncation === true ? :longest_first : Symbol(truncation)
    trunc in (:longest_first, :only_first, :do_not_truncate) ||
        error("`truncation = $(repr(truncation))` must be true, false, :longest_first or :only_first")
    trunc = trunc !== :do_not_truncate
    if max_length === nothing
        limit = tok.model_max_length
        pad === :max_length &&
            (limit === nothing ? (pad = :do_not_pad) : (max_length = limit))
        trunc && (limit === nothing ? (trunc = false) : (max_length = limit))
    end
    return pad, trunc ? max_length : nothing, max_length
end

function (tok::TokenizersBackend)(text; padding = nothing, truncation = nothing,
        max_length::Union{Nothing, Integer} = nothing, padding_side = nothing,
        add_special_tokens::Bool = true)
    texts = is_one_text(text) ? [text] : collect(text)
    all(is_one_text, texts) ||
        error("a text is a string or token ids (from 0), or a vector of them; got $(typeof(text))")
    pad, truncate_to, max_length = call_strategies(tok, padding, truncation, max_length)
    # transformers fails to pad without a padding token, even sequences of one length
    padding === nothing || pad === :do_not_pad || check_pad_token(tok)
    seqs = [token_ids(tok, t, add_special_tokens, truncate_to) for t in texts]
    lengths = length.(seqs)
    targets = pad === :max_length ? max.(lengths, max_length) : lengths
    n = maximum(targets; init = 0)
    if !allequal(targets) && pad !== :longest
        error("texts of $(join(unique(targets), ", ")) tokens make no array: pad them (`padding = true`)" *
              (pad === :max_length ? ", or truncate them (`truncation = true`)" : ""))
    end
    side = padding_side === nothing ? tok.padding_side : String(padding_side)
    side in ("right", "left") || error("`padding_side` must be \"right\" or \"left\"")
    pad_id = all(==(n), lengths) ? 0 : pad_token_id(tok)
    input_ids = fill(pad_id + 1, n, length(seqs))
    attention_mask = zeros(Int, n, length(seqs))
    for (j, ids) in enumerate(seqs)
        r = side == "right" ? (1:length(ids)) : ((n - length(ids) + 1):n)
        input_ids[r, j] .= ids .+ 1
        attention_mask[r, j] .= 1
    end
    if is_one_text(text)
        input_ids, attention_mask = vec(input_ids), vec(attention_mask)
    end
    "attention_mask" in tok.model_input_names || return (; input_ids)
    return (; input_ids, attention_mask)
end

# The keywords of the tokenizer's call, and the values with which the other keywords of
# `TEXT_KWARGS` change nothing in the call of a processor (`stride` and `return_tensors`
# change nothing whatever their value: Metalhead returns no overflowing tokens, and arrays).
const CALL_KEYWORDS = (:padding, :truncation, :max_length, :padding_side,
    :add_special_tokens)
const INERT_TEXT_KWARGS = (; is_split_into_words = false, pad_to_multiple_of = nothing,
    return_token_type_ids = false, return_attention_mask = true,
    return_overflowing_tokens = false, return_special_tokens_mask = false,
    return_offsets_mapping = false, return_length = false,
    return_mm_token_type_ids = true, return_text_replacement_offsets = false)

# Throw if the `init_kwargs` of the tokenizer of the processor `P` set a keyword of its
# calls that Metalhead doesn't implement.
function check_init_kwargs(P, tok)
    tok isa TokenizersBackend || return nothing
    for (k, v) in pairs(tok.init_kwargs)
        (k in CALL_KEYWORDS || k in (:stride, :return_tensors)) && continue
        inert = get(INERT_TEXT_KWARGS, k, missing)
        isequal(v, inert) ||
            error("`$k = $(repr(v))` in the `init_kwargs` of the tokenizer of $(nameof(P)) (from tokenizer_config.json), a default of its calls, is not supported" *
                  (inert === missing ? "" : " (supported: $(repr(inert)))"))
    end
    return nothing
end

# The keywords of the call of the tokenizer in that of a processor `P`, as transformers'
# `_merge_kwargs` merges them: those of the call, `kwargs`, over the `init_kwargs` of the
# tokenizer, over the defaults of the tokenizer's call.
function merged_text_kwargs(P, tok::TokenizersBackend, kwargs)
    for k in keys(kwargs)
        k in CALL_KEYWORDS ||
            throw(ArgumentError("$(nameof(P)) has no keyword `$k` (its keywords of text are $(join(CALL_KEYWORDS, ", ")))"))
    end
    init = (; (k => v for (k, v) in pairs(tok.init_kwargs) if k in CALL_KEYWORDS)...)
    return merge(init, kwargs)
end

"""
    decode(processor, indices; skip_special_tokens = false)

The text of the token indices `indices` (1-based, as the models take and give them) of the
tokenizer of `processor`, a [`TokenizersBackend`](@ref
Metalhead.Transformers.TokenizersBackend) or a processor with one, as transformers'
`decode`: a string for an index or a vector of them, a vector of strings for a matrix
`(seq_len, batch)`. Needs a tokenizer backend (`using HuggingFaceTokenizers`).
"""
function decode(tok::TokenizersBackend, indices::AbstractVector;
        skip_special_tokens::Bool = false)
    all(i -> i isa Integer, indices) ||
        throw(ArgumentError("token indices are integers, got $(eltype(indices))"))
    b = backend(tok)
    ids = Int[i - 1 for i in indices]
    skip_special_tokens && (ids = filter(!in(special_token_ids(tok, b)), ids))
    text = decode_ids(b, ids; skip_special_tokens)
    return clean_decoded(tok, text)
end
function decode(tok::TokenizersBackend, indices::AbstractMatrix; kwargs...)
    return [decode(tok, ids; kwargs...) for ids in eachcol(indices)]
end
decode(tok::TokenizersBackend, index::Integer; kwargs...) = decode(tok, [index]; kwargs...)
function decode(tok::TokenizersBackend, indices; kwargs...)
    throw(ArgumentError("decode takes token indices, as an integer, a vector or a matrix, got a $(typeof(indices))"))
end
function decode(p::Processor, indices; kwargs...)
    haskey(processor_components(typeof(p)), :tokenizer) && p.tokenizer !== nothing ||
        error("$(nameof(typeof(p))) has no tokenizer")
    return decode(p.tokenizer, indices; kwargs...)
end

# The ids of the named special tokens and of the extra special tokens, which transformers'
# `__init__` adds to the backend as special: decoding skips them as those of tokenizer.json.
function special_token_ids(tok::TokenizersBackend, b)
    tokens = [(getfield(tok, k) for k in SPECIAL_TOKENS)...,
        values(tok.model_specific_special_tokens)..., tok.extra_special_tokens...]
    ids = Set{Int}()
    for t in unique(filter(!isnothing, tokens))
        id = find_token_id(tok, t, b)
        id === nothing || push!(ids, id)
    end
    return ids
end

# What the classes of transformers' tokenizers change in the text of their backend: CLIP's
# marks the ends of words with "</w>".
const DECODED_FIXES = Dict{String, Function}(
    "CLIPTokenizer" => text -> strip(replace(text, "</w>" => " ")))

function clean_decoded(tok::TokenizersBackend, text)
    text = String(get(DECODED_FIXES, tok.tokenizer_class, identity)(text))
    tok.clean_up_tokenization_spaces || return text
    # transformers skips the clean-up for BPE tokenizers, which it would corrupt
    tok.tokenizer_file === nothing ||
        tokenizer_json(Metalhead.local_path(tok.tokenizer_file)).model_type != "BPE" ||
        return text
    for (a, b) in (" ." => ".", " ?" => "?", " !" => "!", " ," => ",", " ' " => "'",
        " n't" => "n't", " 'm" => "'m", " 's" => "'s", " 've" => "'ve", " 're" => "'re")
        text = replace(text, a => b)
    end
    return text
end
