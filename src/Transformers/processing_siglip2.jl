# Ported from transformers.models.siglip2.processing_siglip2 (transformers 5.21, Apache-2.0,
# The Hugging Face Team): `Siglip2Processor`, the processor of the NaFlex checkpoints of
# SigLIP 2, with the lowercasing of `Siglip2Tokenizer`. Validated against
# test/references/processors/siglip2.safetensors.

"""
    Siglip2Processor(image_processor, tokenizer)

transformers' processor of the NaFlex checkpoints of SigLIP 2: a
[`Siglip2ImageProcessor`](@ref Metalhead.Transformers.Siglip2ImageProcessor) and the
tokenizer of the repository, HuggingFaceTokenizers.jl's `Tokenizer`, which
[`create_processor`](@ref Metalhead.create_processor) loads from a repository such as
`"google/siglip2-base-patch16-naflex"`. The tokenizer needs `using HuggingFaceTokenizers`
before `create_processor`; without it, `tokenizer` is `nothing`, and the processor takes
images but no text.

    (processor::Siglip2Processor)(; images = nothing, text = nothing, max_length = 64)

The inputs of a [`Siglip2Model`](@ref Metalhead.Transformers.Siglip2Model) from `images`
(one image or a vector of them, see [`Siglip2ImageProcessor`](@ref
Metalhead.Transformers.Siglip2ImageProcessor)) and `text`, as a NamedTuple with the keys of
transformers' `Siglip2Processor`, in Julia's layout (PyTorch's dimensions reversed), which
the model takes as it is (`model(inputs)`):

| Key                    | Size                             | Contents                                                     |
|:---------------------- |:-------------------------------- |:------------------------------------------------------------ |
| `input_ids`            | `(max_length, num_texts)`        | token indices of the texts, 1-based, padded on the right; with `text` |
| `pixel_values`         | `(D, max_num_patches, num_images)` | the patches of the images, resized and normalised, `Float32`; with `images` |
| `pixel_attention_mask` | `(max_num_patches, num_images)`  | 1 for the patches of each image, 0 for the padding; with `images` |
| `spatial_shapes`       | `(2, num_images)`                | the numbers of patches `(height, width)` of each image; with `images` |

where `D = 3 * patch_size^2` (768). `text` is a string or a vector of strings, lowercased
(Julia's `lowercase`), as transformers' `Siglip2Tokenizer` lowercases it, and encoded by the
tokenizer with its special tokens (the end-of-text token after the text). The texts are
padded on the right to `max_length` tokens with the padding token of the tokenizer, and
longer ones are cut to `max_length` tokens, of which the last is the end-of-text token, as
transformers' `Siglip2Processor` pads and truncates them by default (`padding =
"max_length"`, `truncation = True`, `max_length = 64`). There is no `attention_mask`, as the
tokenizers of the checkpoints return none (their `model_input_names`): the model attends to
the padding, as it was trained.

```julia
using Metalhead, Flux, HuggingFaceTokenizers
model = create_model("google/siglip2-base-patch16-naflex")
proc = create_processor("google/siglip2-base-patch16-naflex")
image = rand(UInt8, 640, 480, 3)                       # width × height × channels
inputs = proc(; images = image, text = ["a photo of a cat", "a photo of a dog"])
inputs.spatial_shapes                                  # (14, 18): 14 rows of 18 patches
probs = sigmoid.(model(inputs).logits_per_image)       # 2 × 1
```
"""
@processor (image_processor, tokenizer) struct Siglip2Processor <: AbstractProcessor
    image_processor::Union{Nothing, Siglip2ImageProcessor} = nothing
    tokenizer::Any = nothing
    provenance::NamedTuple = (;)
end

register_processor!("Siglip2Processor", Siglip2Processor)

# The tokens of a text, lowercased as `Siglip2Tokenizer` does, and truncated to
# `max_length` with the end-of-text token last.
function siglip2_tokens(tok, text, max_length)
    ids = encode_ids(tok, lowercase(text))
    length(ids) <= max_length && return ids
    return [ids[1:(max_length - 1)]; ids[end]]
end

function (p::Siglip2Processor)(; images = nothing, text = nothing,
        max_length::Integer = SIGLIP_TEXT_LENGTH)
    max_length > 0 || throw(ArgumentError("`max_length` must be positive, got $max_length"))
    encode(tok, t) = siglip2_tokens(tok, t, max_length)
    return siglip_inputs(p, images, text, max_length, encode)
end
