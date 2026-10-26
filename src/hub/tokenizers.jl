# Metalhead ports no tokenizer. The tokenizer of a checkpoint is HuggingFaceTokenizers.jl's
# `Tokenizer` of the repository's tokenizer.json, which runs Hugging Face's tokenizers
# library; ext/MetalheadHuggingFaceTokenizersExt.jl loads it and implements the functions
# below, through which processors and pipelines use a tokenizer. A type with methods of
# `encode_ids`, `token_id` and `pad_token_id` (a test's) stands in for it. Token ids start
# at 0, as tokenizers give them; the models take token indices, which start at 1.

# The tokenizer of the processor files `files`: their tokenizer.json, at the commit of the
# other files, whose padding token is the `pad_token` of tokenizer_config.json unless
# tokenizer.json sets its padding. `nothing` without HuggingFaceTokenizers.
function load_tokenizer(files::ProcessorFiles)
    ext = Base.get_extension(@__MODULE__, :MetalheadHuggingFaceTokenizersExt)
    ext === nothing && return nothing
    config = read_json(files, "tokenizer_config.json")
    path = source_file_if_exists(files, "tokenizer.json")
    path === nothing &&
        error("$(source_name(files.source)) has no tokenizer.json, the file that HuggingFaceTokenizers loads a tokenizer from")
    pad = config === nothing ? nothing : token_content(get(config, "pad_token", nothing))
    return ext.load_tokenizer(path, pad)
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
