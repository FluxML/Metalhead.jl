# Ported from transformers.models.qwen3_vl.processing_qwen3_vl (transformers 5.20,
# Apache-2.0, The Qwen Team and The Hugging Face Team): `Qwen3VLProcessor`, the processor
# of Qwen3.5, for images and text (not videos), with the expansion of the image
# placeholders. Validated against test/references/processors/qwen3_vl.safetensors.

"""
    Qwen3VLProcessor(image_processor, tokenizer)

transformers' processor of Qwen3-VL and Qwen3.5: a [`Qwen2VLImageProcessor`](@ref
Metalhead.Transformers.Qwen2VLImageProcessor) and the tokenizer of the repository,
HuggingFaceTokenizers.jl's `Tokenizer`, which [`create_processor`](@ref
Metalhead.create_processor) loads from a repository such as `"Qwen/Qwen3.5-0.8B"`. The
tokenizer needs `using HuggingFaceTokenizers` before `create_processor`; without it,
`tokenizer` is `nothing`, and the processor takes images but no text.

    (processor::Qwen3VLProcessor)(; images = nothing, text = nothing)

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

`text` is a string or a vector of strings, which the tokenizer encodes with its special
tokens. One text gives vectors `(seq_len,)`, as the model takes them, so that the logits of
its next token are `model(inputs)[:, end]`. A batch of texts is padded on the right to the
longest, with the padding token of the tokenizer, as transformers pads it with
`padding = true`; the other keywords of transformers' tokenizer (truncation, `max_length`)
aren't ported.

Each image placeholder of the texts, `<|image_pad|>`, stands for the next image, in order
through the texts, and becomes as many image tokens as the image has merged patches,
`prod(image_grid_thw[:, i]) ÷ merge_size^2`, as in transformers. The numbers of
placeholders and of images must be equal (transformers fails on more placeholders than
images, and its model on fewer). The chat template isn't applied: write the placeholders
where it would, `<|vision_start|><|image_pad|><|vision_end|>`.

```julia
using Metalhead, HuggingFaceTokenizers
model = create_model("Qwen/Qwen3.5-0.8B")
proc = create_processor("Qwen/Qwen3.5-0.8B")
image = rand(UInt8, 640, 480, 3)                       # width × height × channels
prompt = "<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>" *
         "Describe this image.<|im_end|>\\n<|im_start|>assistant\\n" *
         "<think>\\n\\n</think>\\n\\n"
inputs = proc(; images = image, text = prompt)        # 300 image tokens for 30 × 40 patches
logits = model(inputs)                                 # vocab_size × seq_len
HuggingFaceTokenizers.decode(proc.tokenizer, [argmax(logits[:, end]) - 1])
```
"""
@processor (image_processor, tokenizer) struct Qwen3VLProcessor <: AbstractProcessor
    image_processor::Union{Nothing, Qwen2VLImageProcessor} = nothing
    tokenizer::Any = nothing
    provenance::NamedTuple = (;)
end

register_processor!("Qwen3VLProcessor", Qwen3VLProcessor; model_types = ["qwen3_5", "qwen3_5_moe"])

function ignored_keys(::Type{Qwen3VLProcessor})
    return (; chat_template = "the chat template, which Metalhead doesn't apply",
        video_processor = "the video processor, as videos are out of scope")
end

# The placeholders of `Qwen3VLProcessor.__init__`, those of a tokenizer without
# `image_token` and `video_token` attributes, as HuggingFaceTokenizers' is.
const QWEN3_VL_IMAGE_TOKEN = "<|image_pad|>"
const QWEN3_VL_VIDEO_TOKEN = "<|video_pad|>"

# The texts with their placeholders `token`, image after image, each expanded to `counts`
# copies of itself, as transformers' `get_text_with_replacements`.
function expand_image_tokens(texts, token, counts)
    n = sum(t -> count(token, t), texts; init = 0)
    n == length(counts) ||
        error("the text has $n image placeholders (`$token`) for $(length(counts)) images: write one where each image goes, as `<|vision_start|>$token<|vision_end|>`")
    k = 0
    return map(texts) do text
        pieces = split(text, token)
        expanded = IOBuffer()
        for (i, piece) in enumerate(pieces)
            print(expanded, piece)
            i < length(pieces) && print(expanded, token^counts[k + i])
        end
        k += length(pieces) - 1
        return String(take!(expanded))
    end
end

function mm_token_type_ids(input_ids, image_id, video_id)
    types = zeros(Int, size(input_ids))
    image_id === nothing || (types[input_ids .== image_id + 1] .= 1)
    video_id === nothing || (types[input_ids .== video_id + 1] .= 2)
    return types
end

function (p::Qwen3VLProcessor)(; images = nothing, text = nothing)
    images === nothing && text === nothing &&
        error("Qwen3VLProcessor needs `images` or `text`")
    images === nothing || p.image_processor !== nothing ||
        error("Qwen3VLProcessor has no image processor for `images`")
    texts, one_text = text === nothing ? (nothing, false) : text_list(text)
    tok = text === nothing ? nothing : tokenizer_of(p)
    image_inputs = images === nothing ? (;) : p.image_processor(images)
    text === nothing && return image_inputs
    if images !== nothing
        merge_length = p.image_processor.merge_size^2
        counts = [prod(g) ÷ merge_length for g in eachcol(image_inputs.image_grid_thw)]
        texts = expand_image_tokens(texts, QWEN3_VL_IMAGE_TOKEN, counts)
    end
    text_inputs = pad_ids([encode_ids(tok, t) for t in texts], pad_token_id(tok))
    one_text && (text_inputs = map(vec, text_inputs))
    types = mm_token_type_ids(text_inputs.input_ids, token_id(tok, QWEN3_VL_IMAGE_TOKEN),
        token_id(tok, QWEN3_VL_VIDEO_TOKEN))
    return merge(text_inputs, (; mm_token_type_ids = types), image_inputs)
end
