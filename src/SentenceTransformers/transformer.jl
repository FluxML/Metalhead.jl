# Ported from sentence_transformers.base.modules.transformer (sentence-transformers 6.2.0,
# Apache-2.0, UKP Lab and the Hugging Face Team): the text of the "feature-extraction" task,
# which loads transformers' AutoModel. Its other tasks, modalities and options (chat
# templates, multi-vector query expansion, flash attention's unpadded inputs) aren't ported.

"""
    Transformer

sentence-transformers' `Transformer` module, the first of a
[`SentenceTransformer`](@ref Metalhead.SentenceTransformers.SentenceTransformer): its
`auto_model`, the ported transformers model that transformers' `AutoModel` loads from the
repository's config.json, without a head (the decoder of Qwen3-Embedding's
`Qwen3ForCausalLM`), and its `tokenizer`, HuggingFaceTokenizers.jl's `Tokenizer` of the
repository's tokenizer.json (a reference to the file before `using HuggingFaceTokenizers`).

`m(features)` adds the `token_embeddings` of the `input_ids` of the NamedTuple `features`,
the last hidden states of `auto_model`, of size `(hidden_size, seq_len, batch)`, with their
`attention_mask`.

Texts are encoded with the special tokens of the tokenizer, truncated to `max_seq_length`
tokens (on the `truncation_side` of tokenizer_config.json), and padded on the right. The
`max_seq_length` of sentence_bert_config.json takes precedence over the `model_max_length`
of tokenizer_config.json (where sentence-transformers 5 saves it), which is capped at the
`max_position_embeddings` of config.json, as sentence-transformers does. `do_lower_case` is
supported where the normalizer of the tokenizer already lowercases.
"""
@concrete struct Transformer
    auto_model
    tokenizer
    max_seq_length
    do_lower_case
    truncation_side
end
Flux.@layer Transformer trainable=(auto_model,)

function (m::Transformer)(features::NamedTuple)
    attention_mask = get(features, :attention_mask, nothing)
    token_embeddings = m.auto_model(features.input_ids; attention_mask)
    return merge(features, (; token_embeddings))
end

# transformers' `AutoModel` of the model types that the Transformer module supports: the
# registered architecture whose base model it is, and the arguments that build it without a
# head of its own (a tied head shares the weights of the embeddings).
const AUTO_MODELS = Dict{String, Tuple{String, NamedTuple}}(
    "qwen3" => ("Qwen3ForCausalLM", (; tie_word_embeddings = true)))

# The base model of a model with a head, and the prefix of its keys in the checkpoints of
# that model (transformers' `base_model_prefix`), which AutoModel strips. A model without a
# head is its own base model, with no prefix unless a method gives one: the keys of a
# checkpoint load with the prefix or without it.
base_model(m) = (m, "")
base_model(m::CausalLM) = decoder(m)

# `AutoModel.from_pretrained` of the config.json `config` of `source`: the base model, whose
# weights are those of a checkpoint of the base model or, under its prefix, of a model with
# a head, whose head isn't loaded.
function auto_model(source, config; pretrained, eltype)
    model_type = get(config, "model_type", nothing)
    haskey(AUTO_MODELS, model_type) ||
        error("the Transformer module of $(source_name(source)) is a transformers model of `model_type = $(repr(model_type))`, whose AutoModel isn't ported to Metalhead for sentence-transformers (supported: $(join(sort!(collect(keys(AUTO_MODELS))), ", ")))")
    architecture, args = AUTO_MODELS[model_type]
    spec = lookup(architecture)
    convert_eltype = ELTYPE_CONVERTERS[eltype]
    args = model_args(TRANSFORMERS, spec, config, args)
    full = pretrained ? build_pretrained(spec.build, args, eltype) :
           convert_eltype(spec.build(; args...))
    model, prefix = base_model(full)
    if pretrained
        ckpt = Checkpoint(source_weights(source))
        ignore = [DEFAULT_IGNORE; spec.ignore]
        if !isempty(prefix) && any(k -> startswith(k, prefix * "."), keys(ckpt))
            load_state!(model, ckpt; prefix, ignore = [ignore; Regex("^(?!$prefix\\.)")])
        else
            load_state!(model, ckpt; ignore)
        end
    end
    return Flux.testmode!(model)
end

const SENTENCE_BERT_CONFIGS = ["sentence_bert_config.json", "sentence_roberta_config.json",
    "sentence_distilbert_config.json", "sentence_camembert_config.json",
    "sentence_albert_config.json", "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json"]

const TEXT_MODALITY = Dict("text" => Dict("method" => "forward",
    "method_output_name" => "last_hidden_state"))

function load_transformer(source; pretrained, eltype)
    found = findfirst(f -> source_file_if_exists(source, f) !== nothing,
        SENTENCE_BERT_CONFIGS)
    file = found === nothing ? first(SENTENCE_BERT_CONFIGS) : SENTENCE_BERT_CONFIGS[found]
    dict = found === nothing ? nothing : read_file(source, file)
    c = read_settings(Transformer, dict, file;
        defaults = (; max_seq_length = nothing, do_lower_case = false,
            transformer_task = "feature-extraction", modality_config = nothing,
            module_output_name = "token_embeddings", processing_kwargs = (;),
            query_length = nothing, document_length = nothing, query_expansion = nothing,
            model_kwargs = (;), processor_kwargs = (;), config_kwargs = (;)),
        renamed = (; model_args = :model_kwargs, tokenizer_args = :processor_kwargs,
            config_args = :config_kwargs),
        ignored = (;
            unpad_inputs = "selects the unpadded inputs of flash attention, which give the same embeddings",
            tokenizer_name_or_path = "names the repository of the tokenizer, deprecated"))
    c.transformer_task == "feature-extraction" ||
        unsupported("transformer_task", c.transformer_task, file, "\"feature-extraction\"")
    c.modality_config === nothing || dict["modality_config"] == TEXT_MODALITY ||
        unsupported("modality_config", dict["modality_config"], file,
            "text through `forward`, as `last_hidden_state`")
    c.module_output_name == "token_embeddings" ||
        unsupported("module_output_name", c.module_output_name, file, "\"token_embeddings\"")
    for k in (:query_length, :document_length, :query_expansion)
        c[k] === nothing ||
            unsupported(k, c[k], file, "null, of a model that isn't multi-vector")
    end
    for k in (:processing_kwargs, :model_kwargs, :processor_kwargs, :config_kwargs)
        isempty(c[k]) || unsupported(k, c[k], file, "none")
    end
    config = read_file(source, "config.json")
    config === nothing &&
        error("the Transformer module of $(source_name(source)) has no config.json")
    model = auto_model(source, config; pretrained, eltype)
    files = ProcessorFiles(source)
    tokenizer = load_tokenizer_or_defer(files)
    tokenizer_config = something(read_json(files, "tokenizer_config.json"),
        Dict{String, Any}())
    padding_side = get(tokenizer_config, "padding_side", "right")
    # rotary positions make the embeddings of a causal model independent of the padding
    padding_side == "right" || padding_side == "left" && model isa DecoderModel ||
        unsupported("padding_side", padding_side, "tokenizer_config.json",
            "\"right\", or \"left\" for a causal language model")
    truncation_side = get(tokenizer_config, "truncation_side", "right")
    truncation_side in ("right", "left") ||
        unsupported("truncation_side", truncation_side, "tokenizer_config.json", "\"right\" and \"left\"")
    c.do_lower_case && !tokenizer_lowercases(files) &&
        unsupported("do_lower_case", true, file, "true only for a tokenizer that lowercases")
    max_length = something(c.max_seq_length,
        max_model_length(get(tokenizer_config, "model_max_length", nothing), config),
        Some(nothing))
    return Transformer(model, tokenizer, max_length, c.do_lower_case, truncation_side)
end
MODULE_LOADERS["Transformer"] = load_transformer

# transformers' `model_max_length` of a tokenizer, `nothing` for its `VERY_LARGE_INTEGER`
# (no limit), capped at the `max_position_embeddings` of the model, as sentence-transformers
# does.
function max_model_length(model_max_length, config)
    n = model_max_length isa Real && model_max_length < 1e18 ? Int(model_max_length) :
        nothing
    p = get(config, "max_position_embeddings", nothing)
    p isa Integer && p != -1 || return n
    return n === nothing ? p : min(n, p)
end

# Whether the normalizer of the tokenizer.json of `files` lowercases, so that
# `do_lower_case` changes nothing. sentence-transformers adds a `Lowercase` before a
# `BertNormalizer` that lowercases, which doesn't change its output.
function tokenizer_lowercases(files)
    tokenizer = read_json(files, "tokenizer.json")
    return tokenizer !== nothing && lowercases(get(tokenizer, "normalizer", nothing))
end
lowercases(::Nothing) = false
function lowercases(n::AbstractDict)
    type = get(n, "type", nothing)
    type == "Lowercase" && return true
    type == "BertNormalizer" && return get(n, "lowercase", true) === true
    return type == "Sequence" && any(lowercases, n["normalizers"])
end

# The tokenizer of a Transformer module, which encoding text needs.
function text_tokenizer(m::Transformer)
    tok = resolve_tokenizer(m.tokenizer)
    tok === nothing || return tok
    m.tokenizer isa DeferredTokenizer &&
        error("the tokenizer of the SentenceTransformer needs HuggingFaceTokenizers.jl to encode texts: run `using HuggingFaceTokenizers`, without loading the model again")
    return error("the SentenceTransformer has no tokenizer, as its repository has no tokenizer.json: pass the inputs of the model, `input_ids` and `attention_mask`")
end

# The token ids of `text` with the special tokens that the tokenizer adds around them, as
# `(front, content, back)`.
function special_tokens_around(tok, text)
    ids = encode_ids(tok, text)
    content = encode_ids(tok, text; add_special_tokens = false)
    n = length(content)
    start = findfirst(k -> ids[(k + 1):(k + n)] == content, 0:(length(ids) - n))
    start === nothing &&
        error("the special tokens that the tokenizer adds aren't around the text, as truncating it needs")
    return ids[1:(start - 1)], content, ids[(start + n):end]
end

# The token ids of `text`, with the special tokens of the tokenizer, truncated to
# `max_length` tokens as transformers truncates (`truncation = "longest_first"`): the text
# is cut on `side` to leave room for the special tokens.
function encode_truncated(tok, text, max_length, side)
    ids = encode_ids(tok, text)
    (max_length === nothing || length(ids) <= max_length) && return ids
    front, content, back = special_tokens_around(tok, text)
    keep = max_length - length(front) - length(back)
    keep >= 0 ||
        error("the tokenizer adds $(length(ids) - length(content)) special tokens, more than `max_seq_length = $max_length`")
    kept = side == "right" ? content[1:keep] : content[(end - keep + 1):end]
    return [front; kept; back]
end

# The length of `prompt` in tokens, as sentence-transformers' `_get_prompt_length`: with the
# special tokens that the tokenizer adds before it, but not the last one it adds after it.
function prompt_length(m::Transformer, tok, prompt)
    n = length(encode_truncated(tok, prompt, m.max_seq_length, m.truncation_side))
    return n - !isempty(last(special_tokens_around(tok, prompt)))
end

"""
    preprocess(transformer::Transformer, texts; prompt = nothing)

The inputs of a
[`SentenceTransformer`](@ref Metalhead.SentenceTransformers.SentenceTransformer) for a
vector of strings `texts`, as sentence-transformers' `preprocess` makes them, with the
string `prompt` prepended to each: their token indices, 1-based, truncated to
`max_seq_length` and padded on the right with the padding token of the tokenizer, of size
`(seq_len, batch)`, as `input_ids`; their `attention_mask`; and with a non-empty prompt its
length in tokens, `prompt_length`, which the pooling leaves out where it doesn't include the
prompt.
"""
function preprocess(m::Transformer, texts::AbstractVector{<:AbstractString};
        prompt::Union{Nothing, AbstractString} = nothing)
    tok = text_tokenizer(m)
    prefixed = prompt === nothing ? texts : [prompt * t for t in texts]
    ids = [encode_truncated(tok, t, m.max_seq_length, m.truncation_side) for t in prefixed]
    inputs = pad_ids(ids, pad_token_id(tok))
    (prompt === nothing || isempty(prompt)) && return inputs
    return merge(inputs, (; prompt_length = prompt_length(m, tok, prompt)))
end
