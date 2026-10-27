# Ported from transformers.models.sam3.processing_sam3 (transformers 5.20, Apache-2.0, The
# Hugging Face Team): `Sam3Processor`, which also stands for `Sam3VideoProcessor`, of which
# only the detector of images is ported. Validated against
# test/references/processors/sam3.safetensors.

"""
    Sam3Processor(image_processor, tokenizer; point_pad_value = -10)

transformers' processor of SAM 3: a [`Sam3ImageProcessor`](@ref
Metalhead.Transformers.Sam3ImageProcessor) and the tokenizer of the repository,
HuggingFaceTokenizers.jl's `Tokenizer`, which [`create_processor`](@ref
Metalhead.create_processor) loads from a repository such as `"unsloth/sam3.1"` (or
`"jetjodh/sam3"`, whose `Sam3VideoProcessor` it loads as a `Sam3Processor`). The tokenizer
needs `using HuggingFaceTokenizers` before `create_processor`; without it, `tokenizer` is
`nothing`, and the processor takes images but no prompts.

    (processor::Sam3Processor)(; images = nothing, text = nothing, input_boxes = nothing,
                               input_boxes_labels = nothing, original_sizes = nothing)

The inputs of a [`Sam3Model`](@ref Metalhead.Transformers.Sam3Model) from `images` (one
image or a vector of them, see [`Sam3ImageProcessor`](@ref
Metalhead.Transformers.Sam3ImageProcessor)), `text` and box prompts, as a NamedTuple with
the keys of transformers' `Sam3Processor`, in Julia's layout (PyTorch's dimensions
reversed), which the model takes as it is (`model(inputs)`):

| Key                  | Size                    | Contents                                                                         |
|:-------------------- |:----------------------- |:-------------------------------------------------------------------------------- |
| `pixel_values`       | `(W, H, 3, batch)`      | the images, resized and normalised, `Float32`; with `images`                     |
| `original_sizes`     | `batch`                 | a tuple of the `(; height, width)` of each image, for the boxes and post-processing; `gpu` leaves it on the CPU |
| `input_ids`          | `(32, batch)`           | token indices of the prompts, 1-based, padded on the right; with a prompt        |
| `attention_mask`     | `(32, batch)`           | 1 for the tokens, 0 for the padding; with a prompt                               |
| `input_boxes`        | `(4, num_boxes, batch)` | box prompts `(cx, cy, w, h)`, normalised to the image; only with box prompts     |
| `input_boxes_labels` | `(num_boxes, batch)`    | 1 positive, 0 negative, `point_pad_value` (-10) padding; only with box prompts   |

  - `text` is a string or a vector of strings, which the tokenizer encodes with its special
    tokens. Prompts are padded on the right to 32 tokens, as `processing_sam3.py` writes,
    with the padding token of the tokenizer, and longer ones are kept whole. Without
    `text`, box prompts have the text "visual", as in transformers, and an entry `nothing`
    of a vector is "visual" where its image has boxes.
  - `input_boxes` holds the boxes `(x1, y1, x2, y2)` of each image in its pixels, a vector
    of boxes per image (or `nothing` for none), and `input_boxes_labels` their labels,
    1 (positive, the default) or 0 (negative). Images with fewer boxes are padded with
    `point_pad_value`. The boxes need the images, or their `original_sizes`,
    `(; height, width)` or `(height, width)`.

Post-process the outputs of the model with [`post_process_instance_segmentation`](@ref),
[`post_process_object_detection`](@ref) or [`post_process_semantic_segmentation`](@ref) of
the processor.

```julia
using Metalhead, HuggingFaceTokenizers
model = create_model("unsloth/sam3.1")
proc = create_processor("unsloth/sam3.1")
image = rand(UInt8, 640, 480, 3)                     # width × height × channels
inputs = proc(; images = image, text = "a cat")
objects = post_process_instance_segmentation(proc, model(inputs); threshold = 0.5,
    target_sizes = inputs.original_sizes)
boxes = proc(; images = image, input_boxes = [[(10, 20, 300, 400)]])  # the text "visual"
```
"""
@processor (image_processor, tokenizer) struct Sam3Processor <: AbstractProcessor
    image_processor::Union{Nothing, Sam3ImageProcessor} = nothing
    tokenizer::Any = nothing
    point_pad_value::Int = -10
    provenance::NamedTuple = (;)
end

register_processor!("Sam3Processor", Sam3Processor; aliases = ["Sam3VideoProcessor"],
    model_types = ["sam3", "sam3_video"])

function ignored_keys(::Type{Sam3Processor})
    return (; target_size = "not read by Sam3Processor's `__call__`",
        video_processor = "the video processor of Sam3VideoProcessor, whose tracker " *
                          "of videos isn't ported")
end

# The length of the prompts of SAM 3, written in processing_sam3.py
const SAM3_TEXT_LENGTH = 32

# The boxes or the labels of each image, `nothing` for none, from nested vectors or an
# array whose last dimension is the image: (4, num_boxes, num_images) boxes,
# (num_boxes, num_images) labels.
function per_image(x::AbstractArray{<:Real, 3})
    return [collect(eachcol(b)) for b in eachslice(x; dims = 3)]
end
per_image(x::AbstractMatrix{<:Real}) = collect.(eachcol(x))
per_image(x) = [b === nothing || isempty(b) ? nothing : collect(b) for b in x]

# The prompts, as `_resolve_text_prompts`: "visual" for boxes without text.
function sam3_texts(text, boxes)
    if text === nothing
        return boxes === nothing || isempty(boxes) ? nothing : ["visual"]
    end
    text isa AbstractString && return [text]
    text isa AbstractVector && all(t -> t isa Union{Nothing, AbstractString}, text) ||
        throw(ArgumentError("`text` is a string or a vector of strings, got a $(typeof(text))"))
    texts = Any[t for t in text]
    if boxes !== nothing
        length(texts) == length(boxes) ||
            error("got $(length(texts)) prompts and the boxes of $(length(boxes)) images")
        texts = [t === nothing && boxes[i] !== nothing ? "visual" : t
                 for (i, t) in enumerate(texts)]
    end
    i = findfirst(isnothing, texts)
    i === nothing || error("prompt $i is `nothing`, without boxes")
    return texts
end

function sam3_boxes(p::Sam3Processor, boxes, labels, original_sizes)
    labels = labels === nothing ?
             [b === nothing ? nothing : ones(Int, length(b)) for b in boxes] :
             per_image(labels)
    length(labels) == length(boxes) ||
        error("got the labels of $(length(labels)) images and the boxes of $(length(boxes))")
    length(original_sizes) == length(boxes) ||
        error("got the boxes of $(length(boxes)) images and the sizes of $(length(original_sizes))")
    pad = p.point_pad_value
    n = maximum(b -> b === nothing ? 0 : length(b), boxes; init = 0)
    xyxy = fill(Float32(pad), 4, n, length(boxes))
    box_labels = fill(pad, n, length(boxes))
    for (j, (image_boxes, s)) in enumerate(zip(boxes, original_sizes))
        image_boxes === nothing && continue
        scale = Float32.((s.width, s.height, s.width, s.height))
        length(labels[j]) == length(image_boxes) ||
            error("image $j has $(length(image_boxes)) boxes and $(length(labels[j])) labels")
        for (i, box) in enumerate(image_boxes)
            length(box) == 4 || error("a box is (x1, y1, x2, y2), got $(repr(box))")
            v = Float32.(collect(box))
            # transformers leaves boxes with a padding coordinate as they are
            xyxy[:, i, j] .= any(==(pad), v) ? v : v ./ scale
            box_labels[i, j] = labels[j][i]
        end
    end
    x1, y1, x2, y2 = (xyxy[i:i, :, :] for i in 1:4)
    input_boxes = cat((x1 .+ x2) ./ 2, (y1 .+ y2) ./ 2, x2 .- x1, y2 .- y1; dims = 1)
    return (; input_boxes, input_boxes_labels = box_labels)
end

# The tokens of the prompts, each padded to 32 tokens, or kept whole if longer: a batch of
# prompts longer than others makes no array, as in transformers.
function sam3_tokens(tok, texts)
    ids = [encode_ids(tok, t) for t in texts]
    lengths = unique(max.(length.(ids), SAM3_TEXT_LENGTH))
    length(lengths) == 1 ||
        error("SAM 3 pads its prompts to $SAM3_TEXT_LENGTH tokens, and can't batch prompts of $(join(lengths, ", ")) tokens: shorten the prompts of more than $SAM3_TEXT_LENGTH tokens, or pass them in separate calls")
    return pad_ids(ids, pad_token_id(tok); max_length = only(lengths))
end

function (p::Sam3Processor)(; images = nothing, text = nothing, input_boxes = nothing,
        input_boxes_labels = nothing, original_sizes = nothing)
    images === nothing || p.image_processor !== nothing ||
        error("Sam3Processor has no image processor for `images`")
    inputs = if images !== nothing
        p.image_processor(images)
    elseif original_sizes !== nothing
        (; original_sizes = Tuple(size_list(original_sizes)))
    else
        input_boxes === nothing ||
            error("box prompts need the images, or their `original_sizes`")
        (;)
    end
    boxes = input_boxes === nothing ? nothing : per_image(input_boxes)
    texts = sam3_texts(text, boxes)
    texts === nothing || (inputs = merge(inputs, sam3_tokens(tokenizer_of(p), texts)))
    if boxes !== nothing
        inputs = merge(inputs,
            sam3_boxes(p, boxes, input_boxes_labels, inputs.original_sizes))
    end
    return inputs
end

for f in (:post_process_object_detection, :post_process_instance_segmentation,
    :post_process_semantic_segmentation)
    @eval function $f(p::Sam3Processor, outputs; kwargs...)
        return $f(p.image_processor, outputs; kwargs...)
    end
end
