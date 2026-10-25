# Ported from transformers.models.sam3.processing_sam3 (transformers 5.20, Apache-2.0, The
# Hugging Face Team): `Sam3Processor`, which also stands for `Sam3VideoProcessor`, of which
# only the detector of images is ported. Validated against
# test/references/processors/sam3.safetensors.

"""
    Sam3Processor(image_processor, tokenizer; point_pad_value = -10)

transformers' processor of SAM 3: a [`Sam3ImageProcessor`](@ref
Metalhead.Transformers.Sam3ImageProcessor) and a [`TokenizersBackend`](@ref
Metalhead.Transformers.TokenizersBackend), which [`create_processor`](@ref
Metalhead.create_processor) loads from a repository such as `"unsloth/sam3.1"` (or
`"jetjodh/sam3"`, whose `Sam3VideoProcessor` it loads as a `Sam3Processor`).

    (processor::Sam3Processor)(; images = nothing, text = nothing, input_boxes = nothing,
                               input_boxes_labels = nothing, original_sizes = nothing)

The inputs of a [`Sam3Model`](@ref Metalhead.Transformers.Sam3Model), as a NamedTuple with
transformers' keys, which the model takes as it is (`model(inputs)`):

  - `pixel_values` and `original_sizes`, from the image processor, of one image or a vector
    of images;
  - `input_ids` and `attention_mask`, of size `(32, num_prompts)` (or longer prompts), the
    token indices of `text` padded to 32 tokens, as `processing_sam3.py` writes: a string or
    a vector of strings, which need a tokenizer backend (`using HuggingFaceTokenizers`), or
    the token ids of a tokenizer (from 0, with its special tokens), a vector or a vector of
    them. Without `text`, box prompts have the text "visual", as in transformers, and an
    entry `nothing` of a vector is "visual" where its image has boxes;
  - `input_boxes` `(4, num_boxes, num_images)` and `input_boxes_labels`
    `(num_boxes, num_images)`, from the boxes `(x1, y1, x2, y2)` of each image in its
    pixels, a vector of boxes per image (or `nothing` for none), normalized to
    `(cx, cy, w, h)`, and
    their labels, 1 (positive, the default) or 0 (negative). Images with fewer boxes are
    padded with `point_pad_value`. The boxes need the images, or their `original_sizes`
    `(height, width)`.

Post-process the outputs of the model with [`post_process_instance_segmentation`](@ref
Metalhead.Transformers.post_process_instance_segmentation),
[`post_process_object_detection`](@ref Metalhead.Transformers.post_process_object_detection)
or [`post_process_semantic_segmentation`](@ref
Metalhead.Transformers.post_process_semantic_segmentation) of the processor.

```julia
using Metalhead
using Metalhead.Transformers: post_process_instance_segmentation
model = create_model("unsloth/sam3.1"; pretrained = true)
proc = create_processor("unsloth/sam3.1")
image = rand(UInt8, 640, 480, 3)                     # width × height × channels
inputs = proc(; images = image, text = [[49406, 320, 2368, 49407]])   # "a cat", as ids
objects = post_process_instance_segmentation(proc, model(inputs); threshold = 0.5,
    target_sizes = inputs.original_sizes)
boxes = proc(; images = image, input_boxes = [[(10, 20, 300, 400)]])  # needs a backend
```
"""
Base.@kwdef struct Sam3Processor <: Processor
    image_processor::Any = nothing
    tokenizer::Any = nothing
    point_pad_value::Int = -10
    provenance::NamedTuple = (;)
end

function Sam3Processor(image_processor, tokenizer; kwargs...)
    return Sam3Processor(; image_processor, tokenizer, kwargs...)
end

register_processor!("Sam3Processor", Sam3Processor; aliases = ["Sam3VideoProcessor"],
    model_types = ["sam3", "sam3_video"])

function processor_components(::Type{Sam3Processor})
    return (; image_processor = Sam3ImageProcessor, tokenizer = TokenizersBackend)
end
function ignored_keys(::Type{Sam3Processor})
    return (; target_size = "not read by Sam3Processor's `__call__`",
        video_processor = "the video processor of Sam3VideoProcessor, whose tracker " *
                          "of videos isn't ported")
end

# The length of the prompts of SAM 3, written in processing_sam3.py
const SAM3_TEXT_LENGTH = 32

# The boxes of each image, `nothing` for none, from nested vectors or an array
# (4, num_boxes, num_images).
function sam3_box_list(boxes::AbstractArray{<:Real, 3})
    return [collect(eachcol(b)) for b in eachslice(boxes; dims = 3)]
end
sam3_box_list(boxes) = [b === nothing || isempty(b) ? nothing : collect(b) for b in boxes]
sam3_label_list(labels::AbstractMatrix{<:Integer}) = collect.(eachcol(labels))
function sam3_label_list(labels)
    return [l === nothing || isempty(l) ? nothing : collect(l) for l in labels]
end

# The prompts, as `_resolve_text_prompts`: "visual" for boxes without text.
function sam3_texts(text, boxes)
    if text === nothing
        return boxes === nothing || isempty(boxes) ? nothing : ["visual"]
    end
    is_one_text(text) && return [text]
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
             sam3_label_list(labels)
    length(labels) == length(boxes) ||
        error("got the labels of $(length(labels)) images and the boxes of $(length(boxes))")
    length(original_sizes) == length(boxes) ||
        error("got the boxes of $(length(boxes)) images and the sizes of $(length(original_sizes))")
    pad = p.point_pad_value
    n = maximum(b -> b === nothing ? 0 : length(b), boxes; init = 0)
    xyxy = fill(Float32(pad), 4, n, length(boxes))
    box_labels = fill(pad, n, length(boxes))
    for (j, (image_boxes, (h, w))) in enumerate(zip(boxes, original_sizes))
        image_boxes === nothing && continue
        length(labels[j]) == length(image_boxes) ||
            error("image $j has $(length(image_boxes)) boxes and $(length(labels[j])) labels")
        for (i, box) in enumerate(image_boxes)
            length(box) == 4 || error("a box is (x1, y1, x2, y2), got $(repr(box))")
            v = Float32.(collect(box))
            # transformers leaves boxes with a padding coordinate as they are
            xyxy[:, i, j] .= any(==(pad), v) ? v : v ./ Float32.((w, h, w, h))
            box_labels[i, j] = labels[j][i]
        end
    end
    x1, y1, x2, y2 = (xyxy[i:i, :, :] for i in 1:4)
    input_boxes = cat((x1 .+ x2) ./ 2, (y1 .+ y2) ./ 2, x2 .- x1, y2 .- y1; dims = 1)
    return (; input_boxes, input_boxes_labels = box_labels)
end

function (p::Sam3Processor)(; images = nothing, text = nothing, input_boxes = nothing,
        input_boxes_labels = nothing, original_sizes = nothing)
    inputs = if images !== nothing
        p.image_processor(images)
    elseif original_sizes !== nothing
        (; original_sizes = size_list(original_sizes))
    else
        input_boxes === nothing ||
            error("box prompts need the images, or their `original_sizes`")
        (;)
    end
    boxes = input_boxes === nothing ? nothing : sam3_box_list(input_boxes)
    texts = sam3_texts(text, boxes)
    if texts !== nothing
        tokens = p.tokenizer(texts; padding = :max_length, max_length = SAM3_TEXT_LENGTH)
        inputs = merge(inputs, tokens)
    end
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
