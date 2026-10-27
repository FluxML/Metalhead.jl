module MetalheadHuggingFaceTokenizersExt

using Metalhead: Metalhead, Functors
using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer
using HuggingFaceTokenizers.PythonCall: pyis, pybuiltins, pyconvert

# not a layer: `gpu`, `f16` and the like leave it as it is
Functors.@leaf Tokenizer

# The tokenizers loaded, by tokenizer.json and padding token
const TOKENIZERS = Dict{Tuple{String, Union{Nothing, String}}, Tokenizer}()
const LOCK = ReentrantLock()

# The tokenizer of the tokenizer.json at `path`, whose padding token is `pad` unless the file
# sets its padding: the tokenizers library keeps the padding token in its padding.
function load_tokenizer(path::AbstractString, pad::Union{Nothing, String})
    return lock(LOCK) do
        get!(TOKENIZERS, (path, pad)) do
            tok = HuggingFaceTokenizers.from_file(Tokenizer, path)
            py = tok.py_tokenizer
            if pad !== nothing && is_none(py.padding)
                id = py.token_to_id(pad)
                if is_none(id)
                    @warn "the padding token $(repr(pad)) of tokenizer_config.json is not in the vocabulary of $path: the tokenizer has no padding token"
                else
                    py.enable_padding(; pad_id = id, pad_token = pad)
                end
            end
            return tok
        end
    end
end

is_none(x) = pyis(x, pybuiltins.None)

function Metalhead.encode_ids(tok::Tokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    encoding = tok.py_tokenizer.encode(String(text); add_special_tokens)
    ids = pyconvert(Vector{Int}, encoding.ids)
    # without the padding of a tokenizer.json that pads to a fixed length
    return ids[pyconvert(Vector{Int}, encoding.attention_mask) .!= 0]
end

function Metalhead.token_id(tok::Tokenizer, token::AbstractString)
    id = tok.py_tokenizer.token_to_id(String(token))
    return is_none(id) ? nothing : pyconvert(Int, id)
end

function Metalhead.pad_token_id(tok::Tokenizer)
    padding = tok.py_tokenizer.padding
    return is_none(padding) ? nothing : pyconvert(Int, padding["pad_id"])
end

# The text of the token indices of the models, which start at 1, by the tokenizer of a
# processor, whose token ids start at 0
function HuggingFaceTokenizers.decode(p::Metalhead.AbstractProcessor,
        indices::AbstractVector{<:Integer}; kwargs...)
    ids = [Int(i) - 1 for i in indices]
    return HuggingFaceTokenizers.decode(Metalhead.tokenizer_of(p), ids; kwargs...)
end
function HuggingFaceTokenizers.decode(p::Metalhead.AbstractProcessor, index::Integer;
        kwargs...)
    return HuggingFaceTokenizers.decode(p, [index]; kwargs...)
end

# The embeddings of texts by a sentence-transformers model, which tokenizes them
function HuggingFaceTokenizers.encode(m::Metalhead.SentenceTransformers.SentenceTransformer,
        texts; kwargs...)
    return Metalhead.SentenceTransformers.encode(m, texts; kwargs...)
end

end # module
