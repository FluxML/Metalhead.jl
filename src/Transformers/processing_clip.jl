# Ported from transformers.models.clip.processing_clip (transformers 5.20, Apache-2.0, The
# Hugging Face Team): `CLIPProcessor`, the `ProcessorMixin` of an image processor and a
# tokenizer. Validated against test/references/processors/clip.safetensors.

"""
    CLIPProcessor(image_processor, tokenizer)

transformers' processor of CLIP: a [`CLIPImageProcessor`](@ref
Metalhead.Transformers.CLIPImageProcessor) and the tokenizer of the repository,
HuggingFaceTokenizers.jl's `Tokenizer`, which [`create_processor`](@ref
Metalhead.create_processor) loads from a repository such as
`"openai/clip-vit-base-patch32"`. The tokenizer needs `using HuggingFaceTokenizers` before
`create_processor`; without it, `tokenizer` is `nothing`, and the processor takes images but
no text.

    (processor::CLIPProcessor)(; text = nothing, images = nothing)

The inputs of a [`CLIPModel`](@ref Metalhead.Transformers.CLIPModel) from `text` and
`images` (one image or a vector of them, see [`CLIPImageProcessor`](@ref
Metalhead.Transformers.CLIPImageProcessor)), as a NamedTuple with the keys of transformers'
`CLIPProcessor`, in Julia's layout (PyTorch's dimensions reversed), which the model takes
as it is (`model(inputs)`), and so do its encoders alone:

| Key              | Size                    | Contents                                                              |
|:---------------- |:----------------------- |:--------------------------------------------------------------------- |
| `input_ids`      | `(seq_len, num_texts)`  | token indices of the texts, 1-based, padded on the right; with `text` |
| `attention_mask` | `(seq_len, num_texts)`  | 1 for the tokens, 0 for the padding; with `text`                      |
| `pixel_values`   | `(W, H, 3, num_images)` | the images, resized, cropped and normalised, `Float32`; with `images` |

`text` is a string or a vector of strings, which the tokenizer encodes with its special
tokens; one string is a batch of one text. A batch of texts is padded on the right to the
longest, with the padding token of the tokenizer, as transformers pads it with
`padding = true`; the other keywords of transformers' tokenizer (truncation, `max_length`)
aren't ported, and texts longer than the `max_position_embeddings` of the model (77 tokens)
are an error of the model.

Zero-shot classification is the softmax of the model's `logits_per_image` over the texts,
the labels in a prompt such as "a photo of a cat":

```julia
using Metalhead, HuggingFaceTokenizers
using Flux: softmax
model = create_model("openai/clip-vit-base-patch32")
proc = create_processor("openai/clip-vit-base-patch32")
image = rand(UInt8, 640, 480, 3)                     # width × height × channels
labels = ["cat", "dog", "car"]
inputs = proc(; text = ["a photo of a \$l" for l in labels], images = image)
probabilities = softmax(model(inputs).logits_per_image; dims = 1)   # 3 × 1
```
"""
@processor (image_processor, tokenizer) struct CLIPProcessor <: AbstractProcessor
    image_processor::Union{Nothing, CLIPImageProcessor} = nothing
    tokenizer::Any = nothing
    provenance::NamedTuple = (;)
end

register_processor!("CLIPProcessor", CLIPProcessor; model_types = ["clip"])

function (p::CLIPProcessor)(; text = nothing, images = nothing)
    text === nothing && images === nothing &&
        error("CLIPProcessor needs `text` or `images`")
    images === nothing || p.image_processor !== nothing ||
        error("CLIPProcessor has no image processor for `images`")
    inputs = (;)
    if text !== nothing
        texts, _ = text_list(text)
        tok = tokenizer_of(p)
        inputs = pad_ids([encode_ids(tok, t) for t in texts], pad_token_id(tok))
    end
    images === nothing && return inputs
    return merge(inputs, p.image_processor(images))
end
