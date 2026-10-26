# Ported from transformers.models.qwen3_vl.processing_qwen3_vl and processing_utils
# (transformers 5.20, Apache-2.0, The Qwen Team and The Hugging Face Team):
# `Qwen3VLProcessor`, the processor of Qwen3.5, for images and text (not videos), with the
# expansion of the image placeholders and the merge of the keywords of the tokenizer of
# `ProcessorMixin.__call__`. Validated against
# test/references/processors/qwen3_vl.safetensors and tokenizers.safetensors.

"""
    Qwen3VLProcessor(image_processor, tokenizer)

transformers' processor of Qwen3-VL and Qwen3.5: a [`Qwen2VLImageProcessor`](@ref
Metalhead.Transformers.Qwen2VLImageProcessor) and a [`TokenizersBackend`](@ref
Metalhead.Transformers.TokenizersBackend), which [`create_processor`](@ref
Metalhead.create_processor) loads from a repository such as `"Qwen/Qwen3.5-0.8B"`.

    (processor::Qwen3VLProcessor)(; images = nothing, text = nothing, padding,
                                  truncation, max_length, padding_side,
                                  add_special_tokens)

The inputs of a Qwen3.5 vision-language model ([`qwen3_5_for_conditional_generation`](@ref
Metalhead.Transformers.qwen3_5_for_conditional_generation)) from `images` (one image or a
vector of them, see [`Qwen2VLImageProcessor`](@ref
Metalhead.Transformers.Qwen2VLImageProcessor)) and `text`, as a NamedTuple with the keys of
transformers' `Qwen3VLProcessor`, in Julia's layout (PyTorch's dimensions reversed), which
the model takes as it is (`model(inputs)`): the outputs of the tokenizer and of the image
processor, merged.

| Key                 | Size               | Contents                                                                 |
|:------------------- |:------------------ |:------------------------------------------------------------------------ |
| `input_ids`         | `(seq_len, batch)` | token indices of the texts, 1-based, padded on the right, with the image placeholders expanded; with `text` |
| `attention_mask`    | `(seq_len, batch)` | 1 for the tokens, 0 for the padding; with `text`                         |
| `mm_token_type_ids` | `(seq_len, batch)` | 1 for image tokens, 2 for video tokens, 0 for the others; with `text`    |
| `pixel_values`      | `(D, num_patches)` | the patches of the images, resized and normalised, `Float32`; with `images` |
| `image_grid_thw`    | `(3, num_images)`  | the numbers of patches `(1, height ÷ patch_size, width ÷ patch_size)` of each image; with `images` |

where `D = 3 * temporal_patch_size * patch_size^2` (1536 for Qwen3.5).

`text` is a string or a vector of strings, which need a tokenizer backend
(`using HuggingFaceTokenizers`), or the token ids of a tokenizer (from 0), a vector or a
vector of them. One text gives vectors `(seq_len,)`, as the model takes them, so that the
logits of its next token are `model(inputs)[:, end]`. The other keywords are the
tokenizer's (see [`TokenizersBackend`](@ref Metalhead.Transformers.TokenizersBackend)),
whose defaults are the `init_kwargs` of the tokenizer, those of tokenizer_config.json (such
as `max_length`), as in transformers.

Each image placeholder of the texts, `<|image_pad|>` (the tokenizer's `image_token`), stands
for the next image, in order through the texts, and becomes as many image tokens as the
image has merged patches, `prod(image_grid_thw[:, i]) ÷ merge_size^2`, as in transformers.
The numbers of placeholders and of images must be equal (transformers fails on more
placeholders than images, and its model on fewer), and the image tokens must survive
truncation. The chat template isn't applied: write the placeholders where it would,
`<|vision_start|><|image_pad|><|vision_end|>`.

```julia
using Metalhead, HuggingFaceTokenizers
model = create_model("Qwen/Qwen3.5-0.8B"; pretrained = true)
proc = create_processor("Qwen/Qwen3.5-0.8B")
image = rand(UInt8, 640, 480, 3)                       # width × height × channels
prompt = "<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>" *
         "Describe this image.<|im_end|>\\n<|im_start|>assistant\\n" *
         "<think>\\n\\n</think>\\n\\n"
inputs = proc(; images = image, text = prompt)        # 300 image tokens for 30 × 40 patches
logits = model(inputs)                                 # vocab_size × seq_len
Metalhead.Transformers.decode(proc, [argmax(logits[:, end])])
```
"""
@processor (image_processor, tokenizer) struct Qwen3VLProcessor <: Processor
    image_processor::Any = nothing
    tokenizer::Any = nothing
    provenance::NamedTuple = (;)
end

register_processor!("Qwen3VLProcessor", Qwen3VLProcessor; model_types = ["qwen3_5"])

function processor_components(::Type{Qwen3VLProcessor})
    return (; image_processor = Qwen2VLImageProcessor, tokenizer = TokenizersBackend)
end
function ignored_keys(::Type{Qwen3VLProcessor})
    return (; chat_template = CHAT_KEY,
        video_processor = "the video processor, as videos are out of scope")
end

function check_processor(p::Qwen3VLProcessor)
    Metalhead.check_components(p)
    return check_init_kwargs(Qwen3VLProcessor, p.tokenizer)
end

# `Qwen3VLProcessor.__init__` takes these tokens from the tokenizer, else their defaults.
function named_token(tok::TokenizersBackend, name, default)
    return String(get(tok.model_specific_special_tokens, name, default))
end

count_placeholders(text::AbstractString, token, _) = count(token, text)
count_placeholders(ids::AbstractVector{<:Integer}, _, id) = count(==(id), ids)

# A text with its placeholders, the next of which is image `k + 1`, expanded to `counts`
# copies of themselves, as transformers' `get_text_with_replacements`.
function expand_placeholders(text::AbstractString, token, _, counts, k)
    pieces = split(text, token)
    expanded = IOBuffer()
    for (i, piece) in enumerate(pieces)
        print(expanded, piece)
        i < length(pieces) && print(expanded, token^counts[k + i])
    end
    return String(take!(expanded)), k + length(pieces) - 1
end
function expand_placeholders(ids::AbstractVector{<:Integer}, _, id, counts, k)
    expanded = Int[]
    for t in ids
        t == id ? append!(expanded, fill(id, counts[k += 1])) : push!(expanded, t)
    end
    return expanded, k
end

function expand_image_tokens(texts, token, id, counts)
    n = sum(t -> count_placeholders(t, token, id), texts; init = 0)
    n == length(counts) ||
        error("the text has $n image placeholders (`$token`) for $(length(counts)) images: write one where each image goes, as `<|vision_start|>$token<|vision_end|>`")
    k = 0
    return map(texts) do text
        expanded, k = expand_placeholders(text, token, id, counts, k)
        return expanded
    end
end

# transformers' `_check_special_mm_tokens`: truncation could cut image tokens.
function check_image_tokens(texts, input_ids, token, id)
    columns = input_ids isa AbstractVector ? [input_ids] : eachcol(input_ids)
    expected = [count_placeholders(t, token, id) for t in texts]
    found = [count(==(id + 1), c) for c in columns]
    expected == found ||
        error("the texts have $expected image tokens, but their `input_ids` $found: don't truncate them, or increase `max_length`")
    return nothing
end

function mm_token_type_ids(input_ids, image_id, video_id)
    types = Int.(input_ids .== image_id + 1)
    video_id === nothing || (types[input_ids .== video_id + 1] .= 2)
    return types
end

function (p::Qwen3VLProcessor)(; images = nothing, text = nothing, kwargs...)
    images === nothing && text === nothing &&
        error("Qwen3VLProcessor needs `images` or `text`")
    images === nothing || p.image_processor !== nothing ||
        error("Qwen3VLProcessor has no image processor for `images`")
    tok = p.tokenizer
    text === nothing || tok !== nothing ||
        error("Qwen3VLProcessor has no tokenizer to encode `text`")
    text_kwargs = tok === nothing ? (;) :
                  merged_text_kwargs(Qwen3VLProcessor, tok, NamedTuple(kwargs))
    image_inputs = images === nothing ? (;) : p.image_processor(images)
    text === nothing && return image_inputs
    one_text = is_one_text(text)
    texts = one_text ? [text] : collect(text)
    token = named_token(tok, :image_token, "<|image_pad|>")
    id = token_id(tok, token)
    if images !== nothing
        merge_length = p.image_processor.merge_size^2
        counts = [prod(g) ÷ merge_length for g in eachcol(image_inputs.image_grid_thw)]
        texts = expand_image_tokens(texts, token, id, counts)
    end
    text_inputs = tok(one_text ? only(texts) : texts; text_kwargs...)
    check_image_tokens(texts, text_inputs.input_ids, token, id)
    video_id = find_token_id(tok, named_token(tok, :video_token, "<|video_pad|>"))
    types = mm_token_type_ids(text_inputs.input_ids, id, video_id)
    return merge(text_inputs, (; mm_token_type_ids = types), image_inputs)
end
