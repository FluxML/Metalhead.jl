module MetalheadHuggingFaceTokenizersExt

using Metalhead: Metalhead, create_processor
using Metalhead.Transformers: Transformers, TokenizersBackend
using Metalhead.Diffusers: Flux2KleinPipeline, chat_prompt
using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer
using HuggingFaceTokenizers.PythonCall: pyis, pybuiltins, pyimport, pyconvert, pylist

# The backend of a TokenizersBackend: HuggingFaceTokenizers' tokenizer of its
# tokenizer.json, loaded once per file.
const TOKENIZERS = Dict{String, Tokenizer}()
const LOCK = ReentrantLock()

function load_tokenizer(file::AbstractString)
    return lock(LOCK) do
        get!(TOKENIZERS, file) do
            tokenizer = HuggingFaceTokenizers.from_file(Tokenizer, file)
            py = tokenizer.py_tokenizer
            # TokenizersBackend pads and truncates; transformers takes these as defaults
            pyis(py.padding, pybuiltins.None) && pyis(py.truncation, pybuiltins.None) ||
                error("$file pads or truncates, which Metalhead doesn't read")
            add_config_tokens!(py, joinpath(dirname(file), "tokenizer_config.json"))
            return tokenizer
        end
    end
end

# The tokens of the `added_tokens_decoder` of tokenizer_config.json that tokenizer.json
# lacks, added in the order of their ids as transformers' `TokenizersBackend.__init__` adds
# them (Qwen3.5's audio tokens), which must then have the ids of the configuration.
function add_config_tokens!(py, config_file)
    isfile(config_file) || return nothing
    decoder = get(Metalhead.JSON.parsefile(config_file), "added_tokens_decoder", nothing)
    decoder === nothing && return nothing
    absent = sort!([(parse(Int, id), t) for (id, t) in decoder
                     if pyis(py.token_to_id(t["content"]), pybuiltins.None)]; by = first)
    isempty(absent) && return nothing
    AddedToken = pyimport("tokenizers").AddedToken
    flags(t) = (; (Symbol(k) => Bool(t[k])
                   for k in ("single_word", "lstrip", "rstrip", "normalized", "special")
                   if haskey(t, k))...)
    py.add_tokens(pylist([AddedToken(t["content"]; flags(t)...) for (_, t) in absent]))
    for (id, t) in absent
        added = pyconvert(Int, py.token_to_id(t["content"]))
        added == id ||
            error("$(t["content"]) of the `added_tokens_decoder` of $config_file has the id $id, but $added in its tokenizer.json")
    end
    return nothing
end

function Transformers.encode_ids(t::Tokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    return HuggingFaceTokenizers.encode(t, String(text); add_special_tokens).ids
end

function Transformers.decode_ids(t::Tokenizer, ids::AbstractVector{<:Integer};
        skip_special_tokens::Bool = false)
    return HuggingFaceTokenizers.decode(t, collect(Int, ids); skip_special_tokens)
end

function Metalhead.Diffusers.tokenize(p::Flux2KleinPipeline,
        prompts::AbstractVector{<:AbstractString}; max_sequence_length::Integer = 512)
    p.tokenizer === nothing &&
        error("the pipeline has no tokenizer; pass prompt embeddings instead")
    tokenizer = create_processor(p.tokenizer)
    tokenizer isa TokenizersBackend || error("$(p.tokenizer) holds no tokenizer")
    out = tokenizer(chat_prompt.(prompts); padding = :max_length, truncation = true,
        max_length = max_sequence_length)
    return out.input_ids, out.attention_mask
end

end # module
