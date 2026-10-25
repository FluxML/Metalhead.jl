# Ported from transformers.tokenization_utils_tokenizers and tokenization_utils_base
# (transformers 5.20, Apache-2.0, The Hugging Face Team): `TokenizersBackend`, the tokenizer
# of transformers 5 for a tokenizer.json, with the settings of tokenizer_config.json
# (special tokens, padding, truncation) read as `from_pretrained` reads them. Tokenizing a
# string, a tokenization algorithm, is left to a backend: the HuggingFaceTokenizers.jl
# extension, or any type with methods of `encode_ids` and `decode_ids`. Validated against
# test/references/processors/sam3.safetensors.

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

transformers' `TokenizersBackend`, the tokenizer of a tokenizer.json, which
[`create_processor`](@ref Metalhead.create_processor) loads from a repository with its
settings in tokenizer_config.json (with special_tokens_map.json for files without
`added_tokens_decoder`, as transformers reads them). Its fields are transformers'
attributes:

  - `tokenizer`: the backend that encodes and decodes strings (`encode_ids` and
    `decode_ids`), or `nothing` until a string needs it, when the HuggingFaceTokenizers.jl
    extension loads `tokenizer_file`, the repository's tokenizer.json;
  - `tokenizer_class`: the class that tokenizer_config.json names;
  - the special tokens `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`,
    `cls_token` and `mask_token`, `extra_special_tokens`, and the named tokens of a model
    (`model_specific_special_tokens`, such as `image_token`);
  - `added_tokens_decoder`: the content of the added tokens by id (from 0), from which the
    ids of special tokens are known without a backend;
  - `model_max_length` (`nothing` for no limit), `padding_side`, `truncation_side`,
    `clean_up_tokenization_spaces` and `model_input_names`.

Call it as `tok(text; padding, truncation, max_length, padding_side, add_special_tokens)`
on a string or on token ids from 0, as a tokenizer gives them (which need no backend), or on
a vector of them. It returns `(; input_ids, attention_mask)`: vectors for one text, else
matrices `(seq_len, batch)`, with 1-based token indices, as the models take them. The
keywords are transformers', with its defaults: `padding` is `true` (or `:longest`),
`:max_length` or `false`; `truncation` (`true`, or implied by `max_length` without padding)
cuts the tokens of a string, keeping the special tokens; `max_length` defaults to
`model_max_length`. Without `padding`, a batch of different lengths is padded to the
longest, as a Julia array needs, where transformers would fail to make a tensor of it.
[`decode`](@ref Metalhead.Transformers.decode) turns indices back into text.
"""
Base.@kwdef struct TokenizersBackend <: Processor
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
    model_input_names::Vector{String} = ["input_ids", "attention_mask"]
    provenance::NamedTuple = (;)
end

register_processor!("TokenizersBackend", TokenizersBackend)

const SPECIAL_TOKENS = (:bos_token, :eos_token, :unk_token, :sep_token, :pad_token,
    :cls_token, :mask_token)

const TOKENIZER_JSON_KEY = "configures the tokenizer that transformers rebuilds, which " *
                           "the backend reads from tokenizer.json"
const CHAT_KEY = "chat templates are out of scope"

function ignored_keys(::Type{TokenizersBackend})
    return (; chat_template = CHAT_KEY, use_default_system_prompt = CHAT_KEY,
        response_template = CHAT_KEY, response_schema = CHAT_KEY,
        chat_control_tokens = CHAT_KEY,
        add_bos_token = "dropped by transformers when tokenizer.json exists",
        add_eos_token = "dropped by transformers when tokenizer.json exists",
        add_prefix_space = TOKENIZER_JSON_KEY, do_lower_case = TOKENIZER_JSON_KEY,
        errors = TOKENIZER_JSON_KEY, pretokenize_regex = TOKENIZER_JSON_KEY,
        legacy = TOKENIZER_JSON_KEY, sp_model_kwargs = TOKENIZER_JSON_KEY,
        strip_accents = TOKENIZER_JSON_KEY, tokenize_chinese_chars = TOKENIZER_JSON_KEY,
        do_basic_tokenize = TOKENIZER_JSON_KEY, never_split = TOKENIZER_JSON_KEY,
        vocab_size = "the size of the vocabulary of tokenizer.json",
        spaces_between_special_tokens = "used by transformers' slow tokenizers only",
        max_length = "not read by transformers' tokenizers (model_max_length is the limit)",
        stride = "not read by transformers' tokenizers",
        truncation_strategy = "not read by transformers' tokenizers",
        pad_to_multiple_of = "not read by transformers' tokenizers",
        pad_token_type_id = "the type id of padding tokens, which Metalhead doesn't return",
        name_or_path = "the name of the repository",
        tokenizer_file = "a file of the repository",
        vocab_file = "a file of the repository", merges_file = "a file of the repository",
        verbose = "logging",
        backend = "the library of the tokenizer", files_loaded = "the files loaded")
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
# a list of extra special tokens), and added_tokens.json.
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
    # `__init__` takes `additional_special_tokens` only without `extra_special_tokens`
    if haskey(kw, "additional_special_tokens") && !haskey(kw, "extra_special_tokens")
        set!("extra_special_tokens", kw["additional_special_tokens"]...)
    end
    entries = Tuple{String, Any, String}[]
    named = Pair{Symbol, String}[]
    for k in order
        haskey(kw, k) || continue
        v, file = kw[k]
        if k in ("additional_special_tokens", "added_tokens_decoder")
            continue
        elseif k == "extra_special_tokens"
            if v isa AbstractDict
                append!(named, Symbol(n) => token_content(t) for (n, t) in v)
            elseif !falsy(v)
                push!(entries, (k, token_content.(v), file))
            end
        elseif Symbol(k) in SPECIAL_TOKENS
            push!(entries, (k, token_content(v), file))
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
    return entries
end

function load_processor(::Type{TokenizersBackend}, files::ProcessorFiles; kwargs...)
    has_file(files, "tokenizer_config.json") || has_file(files, "tokenizer.json") ||
        error("$(Metalhead.source_name(files.source)) has no tokenizer (tokenizer_config.json or tokenizer.json)")
    entries = tokenizer_entries(files)
    entries = [entries;
               ("tokenizer_file", SourceFile(files.source, "tokenizer.json"), "repository")]
    for i in eachindex(entries)
        k, v, file = entries[i]
        k == "tokenizer_class" && v isa AbstractString &&
            (entries[i] = (k, Metalhead.class_name(v), file))
    end
    return processor_from_entries(TokenizersBackend, entries; kwargs...)
end

function Metalhead.setting_origin(tok::TokenizersBackend, key, value)
    key === :tokenizer && value === nothing && tok.tokenizer_file !== nothing &&
        return "loaded from tokenizer_file when a string needs it"
    return Metalhead.origin_text(get(tok.provenance, key, nothing), value)
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
    return Base.invokelatest(ext.load_tokenizer, Metalhead.local_path(tok.tokenizer_file))
end

# The id (from 0) of a token: from the added tokens, else from the backend.
function token_id(tok::TokenizersBackend, token)
    for id in sort!(collect(keys(tok.added_tokens_decoder)))
        tok.added_tokens_decoder[id] == token && return id
    end
    ids = encode_ids(backend(tok), token; add_special_tokens = false)
    length(ids) == 1 || error("`$token` is not a token of $(tok.tokenizer_class)")
    return only(ids)
end

function pad_token_id(tok::TokenizersBackend)
    tok.pad_token === nothing &&
        error("$(tok.tokenizer_class) has no padding token (`pad_token`) to pad sequences of different lengths")
    return token_id(tok, tok.pad_token)
end

# The ids of `text`, truncated to `max_length` (unless `nothing`) as the tokenizers library
# truncates: the tokens of the text, without those that the post-processor adds.
function token_ids(tok::TokenizersBackend, text::AbstractString, add_special_tokens,
        max_length)
    b = backend(tok)
    ids = collect(Int, encode_ids(b, text; add_special_tokens))
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

"""
    decode(processor, indices; skip_special_tokens = false)

The text of the token indices `indices` (1-based, as the models take and give them) of the
tokenizer of `processor`, a [`TokenizersBackend`](@ref
Metalhead.Transformers.TokenizersBackend) or a processor with one, as transformers'
`decode`: a string for a vector of indices, a vector of strings for a matrix
`(seq_len, batch)`. Needs a tokenizer backend (`using HuggingFaceTokenizers`).
"""
function decode(tok::TokenizersBackend, indices::AbstractVector{<:Integer};
        skip_special_tokens::Bool = false)
    text = decode_ids(backend(tok), collect(Int, indices) .- 1; skip_special_tokens)
    return clean_decoded(tok, text)
end
function decode(tok::TokenizersBackend, indices::AbstractMatrix{<:Integer}; kwargs...)
    return [decode(tok, ids; kwargs...) for ids in eachcol(indices)]
end
function decode(p::Processor, indices; kwargs...)
    haskey(processor_components(typeof(p)), :tokenizer) ||
        error("$(nameof(typeof(p))) has no tokenizer")
    return decode(p.tokenizer, indices; kwargs...)
end

# What the classes of transformers' tokenizers change in the text of their backend: CLIP's
# marks the ends of words with "</w>".
const DECODED_FIXES = Dict{String, Function}(
    "CLIPTokenizer" => text -> strip(replace(text, "</w>" => " ")))

const MODEL_TYPES = Dict{String, String}()

# The type of the model of tokenizer.json (BPE, WordPiece, ...), read once.
function tokenizer_model_type(tok::TokenizersBackend)
    tok.tokenizer_file === nothing && return "unknown"
    path = Metalhead.local_path(tok.tokenizer_file)
    return get!(MODEL_TYPES, path) do
        return string(get(JSON.parsefile(path)["model"], "type", "unknown"))
    end
end

function clean_decoded(tok::TokenizersBackend, text)
    text = String(get(DECODED_FIXES, tok.tokenizer_class, identity)(text))
    # transformers skips the clean-up for BPE tokenizers, which it would corrupt
    tok.clean_up_tokenization_spaces && tokenizer_model_type(tok) != "BPE" || return text
    for (a, b) in (" ." => ".", " ?" => "?", " !" => "!", " ," => ",", " ' " => "'",
        " n't" => "n't", " 'm" => "'m", " 's" => "'s", " 've" => "'ve", " 're" => "'re")
        text = replace(text, a => b)
    end
    return text
end
