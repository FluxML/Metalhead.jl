# Ported from transformers.models.siglip.processing_siglip (transformers 5.21, Apache-2.0,
# The Hugging Face Team): `SiglipProcessor`, the processor of SigLIP and of the SigLIP 2
# checkpoints of fixed resolution. Validated against
# test/references/processors/siglip.safetensors.

"""
    SiglipProcessor(image_processor, tokenizer)

transformers' processor of SigLIP: a [`SiglipImageProcessor`](@ref
Metalhead.Transformers.SiglipImageProcessor) and the tokenizer of the repository,
HuggingFaceTokenizers.jl's `Tokenizer`, which [`create_processor`](@ref
Metalhead.create_processor) loads from a repository such as
`"google/siglip-base-patch16-224"` or `"google/siglip2-base-patch16-224"`. The tokenizer
needs `using HuggingFaceTokenizers` before `create_processor`; without it, `tokenizer` is
`nothing`, and the processor takes images but no text.

    (processor::SiglipProcessor)(; images = nothing, text = nothing, max_length = 64)

The inputs of a [`SiglipModel`](@ref Metalhead.Transformers.SiglipModel) from `images` (one
image or a vector of them, see [`SiglipImageProcessor`](@ref
Metalhead.Transformers.SiglipImageProcessor)) and `text`, as a NamedTuple with the keys of
transformers' `SiglipProcessor`, in Julia's layout (PyTorch's dimensions reversed), which
the model takes as it is (`model(inputs)`):

| Key            | Size                     | Contents                                                       |
|:-------------- |:------------------------ |:-------------------------------------------------------------- |
| `input_ids`    | `(max_length, num_texts)` | token indices of the texts, 1-based, padded on the right; with `text` |
| `pixel_values` | `(W, H, 3, num_images)`  | the images, resized and normalised, `Float32`; with `images`  |

`text` is a string or a vector of strings, which the tokenizer encodes with its special
tokens (the end-of-text token after the text), padded on the right to `max_length` tokens
with the padding token of the tokenizer, as SigLIP was trained: as transformers'
`SiglipProcessor` pads them with `padding = "max_length"`, which the model cards pass
(transformers pads none by default). The 64 tokens of the default are the length of the
text models of the checkpoints, but for `google/siglip-so400m-patch14-224`, whose text model
takes 16 (`max_length = 16`). A longer text is an error, as transformers doesn't truncate
the texts of SigLIP. There is no `attention_mask`, as the tokenizers of the checkpoints
return none (their `model_input_names`): the model attends to the padding, as it was
trained.

The tokenizer of the SigLIP checkpoints (`SiglipTokenizer`) lowercases the texts and drops
their punctuation before encoding them; their tokenizer.json does the same, but keeps the
characters `/`, `<` and `>`, so that a text with them gets other tokens than in
transformers. The SigLIP 2 checkpoints of fixed resolution (Gemma's tokenizer) give the
same tokens as transformers.

```julia
using Metalhead, Flux, HuggingFaceTokenizers
model = create_model("google/siglip-base-patch16-224")
proc = create_processor("google/siglip-base-patch16-224")
image = rand(UInt8, 640, 480, 3)                       # width × height × channels
inputs = proc(; images = image, text = ["a photo of a cat", "a photo of a dog"])
probs = sigmoid.(model(inputs).logits_per_image)       # 2 × 1
```
"""
@processor (image_processor, tokenizer) struct SiglipProcessor <: AbstractProcessor
    image_processor::Union{Nothing, SiglipImageProcessor} = nothing
    tokenizer::Any = nothing
    provenance::NamedTuple = (;)
end

register_processor!("SiglipProcessor", SiglipProcessor)

# The length of the text models of SigLIP, the `model_max_length` of its tokenizers and
# the `max_length` of `Siglip2ProcessorKwargs`, to which the texts are padded
const SIGLIP_TEXT_LENGTH = 64

# The inputs of the images and texts of a processor of SigLIP or SigLIP 2, with the token
# ids `encode(tok, text)` of each text padded to `max_length`.
function siglip_inputs(p, images, text, max_length, encode)
    images === nothing && text === nothing &&
        error("$(nameof(typeof(p))) needs `images` or `text`")
    images === nothing || p.image_processor !== nothing ||
        error("$(nameof(typeof(p))) has no image processor for `images`")
    inputs = images === nothing ? (;) : p.image_processor(images)
    text === nothing && return inputs
    texts, _ = text_list(text)
    tok = tokenizer_of(p)
    ids = [encode(tok, t) for t in texts]
    padded = pad_ids(ids, pad_token_id(tok); max_length)
    return merge((; padded.input_ids), inputs)
end

function (p::SiglipProcessor)(; images = nothing, text = nothing,
        max_length::Integer = SIGLIP_TEXT_LENGTH)
    return siglip_inputs(p, images, text, max_length, encode_ids)
end
