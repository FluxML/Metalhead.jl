module MetalheadHuggingFaceTokenizersExt

using Metalhead: Metalhead, JSON
using Metalhead.Diffusers: Flux2KleinPipeline, chat_prompt
using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer

const TOKENIZERS = Dict{String, Any}()

# The tokenizer of `dir` and the id of its padding token, loaded once.
function load_tokenizer(dir)
    return get!(TOKENIZERS, dir) do
        tokenizer = HuggingFaceTokenizers.from_file(Tokenizer, joinpath(dir, "tokenizer.json"))
        pad = JSON.parsefile(joinpath(dir, "tokenizer_config.json"))["pad_token"]
        pad = pad isa AbstractDict ? pad["content"] : pad
        return tokenizer, only(HuggingFaceTokenizers.encode(tokenizer, pad).ids)
    end
end

function Metalhead.Diffusers.tokenize(p::Flux2KleinPipeline,
        prompts::AbstractVector{<:AbstractString}; max_sequence_length::Integer = 512)
    p.tokenizer === nothing &&
        error("the pipeline has no tokenizer; pass prompt embeddings instead")
    tokenizer, pad = load_tokenizer(p.tokenizer)
    ids = fill(pad, max_sequence_length, length(prompts))
    mask = zeros(Int, max_sequence_length, length(prompts))
    for (j, prompt) in enumerate(prompts)
        tokens = HuggingFaceTokenizers.encode(tokenizer, chat_prompt(prompt)).ids
        n = min(length(tokens), max_sequence_length)
        ids[1:n, j] = tokens[1:n]
        mask[1:n, j] .= 1
    end
    return ids .+ 1, mask
end

end # module
