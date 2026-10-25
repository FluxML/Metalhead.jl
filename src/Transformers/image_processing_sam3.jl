# Ported from transformers.models.sam3.image_processing_sam3 and processing_sam3
# (transformers 5.20, Apache-2.0, The Hugging Face Team): the inputs that `Sam3Processor`
# makes of images, box prompts and token ids, and its post-processing of the outputs of
# SAM 3. Validated against test/references/sam3_image_processor.safetensors.

"""
    sam3_image_inputs(images; image_size = (1008, 1008), image_mean = (0.5, 0.5, 0.5),
                      image_std = (0.5, 0.5, 0.5)) -> (pixel_values, original_sizes)

The images of SAM 3 ([`sam3_model`](@ref Metalhead.Transformers.sam3_model)), as
transformers' image processor of SAM 3 makes them: each image is resized to `image_size`
`(height, width)` with antialiased bilinear interpolation (an 8-bit image along its width
and then its height, rounded back to integers after each, as torchvision does on the CPU),
scaled to `[0, 1]` (8-bit images only) and normalized with `image_mean` and `image_std`.

`images` is one image or a vector of images, each an array of size
`(width, height, channels)` (torch's `(channels, height, width)` reversed) of integers from
0 to 255 (an 8-bit image, as `UInt8`) or of real numbers from 0 to 1. For an Images.jl
image `img`, that is `permutedims(channelview(img), (3, 2, 1))`.

Returns `pixel_values`, a `Float32` array of size `(width, height, channels, num_images)`,
and `original_sizes`, the `(height, width)` of each image, which
[`sam3_box_inputs`](@ref Metalhead.Transformers.sam3_box_inputs) and the post-processing
(the `target_sizes` of [`post_process_instance_segmentation`](@ref
Metalhead.Transformers.post_process_instance_segmentation)) take.

An 8-bit image whose size needs no resizing gives the same `pixel_values` as transformers,
and an image of real numbers the same to Float32 precision. A resized 8-bit image may differ
by one level of 255 in a few of its values, as torchvision computes its weights in
fixed-point arithmetic.
"""
function sam3_image_inputs(images; image_size = (1008, 1008),
        image_mean = (0.5, 0.5, 0.5), image_std = (0.5, 0.5, 0.5))
    images isa AbstractArray{<:Real, 3} &&
        return sam3_image_inputs([images]; image_size, image_mean, image_std)
    h, w = image_size isa Integer ? (image_size, image_size) : image_size
    pixel_values = map(images) do image
        ndims(image) == 3 ||
            error("an image must be an array of size (width, height, channels), got $(size(image))")
        size(image, 3) == length(image_mean) == length(image_std) ||
            error("an image has $(size(image, 3)) channels, `image_mean` and `image_std` $(length(image_mean)) and $(length(image_std))")
        # transformers folds the rescaling of 8-bit images into the mean and the std
        scale = eltype(image) <: Integer ? 255.0f0 : 1.0f0
        mean = reshape(Float32.(collect(image_mean)) .* scale, 1, 1, :)
        std = reshape(Float32.(collect(image_std)) .* scale, 1, 1, :)
        resized = resize_image(image, w, h; upsample = NNlib.upsample_bilinear)
        return (resized .- mean) ./ std
    end
    return stack(pixel_values), [(size(image, 2), size(image, 1)) for image in images]
end

"""
    sam3_box_inputs(boxes, original_sizes; labels = nothing)
        -> (input_boxes, input_boxes_labels)

The box prompts of SAM 3, as transformers' processor of SAM 3 makes them: `boxes` holds,
for each image, a vector of boxes `(x1, y1, x2, y2)` in the pixels of the image, whose
size `(height, width)` is in `original_sizes`, and `labels` their labels, 1 (positive, the
default) or 0 (negative).

Returns `input_boxes`, of size `(4, num_boxes, num_images)`, the boxes `(cx, cy, w, h)`
normalized to their image, and `input_boxes_labels`, of size `(num_boxes, num_images)`; an
image with fewer boxes than the others is padded with boxes labelled -10.
"""
function sam3_box_inputs(boxes, original_sizes; labels = nothing)
    length(boxes) == length(original_sizes) ||
        error("got the boxes of $(length(boxes)) images and the sizes of $(length(original_sizes))")
    n = maximum(length, boxes; init = 0)
    xyxy = fill(-10.0f0, 4, n, length(boxes))
    box_labels = fill(-10, n, length(boxes))
    for (j, (image_boxes, (h, w))) in enumerate(zip(boxes, original_sizes))
        for (i, (x1, y1, x2, y2)) in enumerate(image_boxes)
            xyxy[:, i, j] .= Float32.((x1, y1, x2, y2)) ./ Float32.((w, h, w, h))
            box_labels[i, j] = labels === nothing ? 1 : labels[j][i]
        end
    end
    x1, y1, x2, y2 = (xyxy[i:i, :, :] for i in 1:4)
    return cat((x1 .+ x2) ./ 2, (y1 .+ y2) ./ 2, x2 .- x1, y2 .- y1; dims = 1), box_labels
end

"""
    sam3_text_inputs(token_ids; max_length = 32, pad_token_id = 49407)
        -> (input_ids, attention_mask)

The text prompts of SAM 3, as transformers' processor of SAM 3 makes them: the token ids of
each prompt, 0-based as a tokenizer gives them (with its start and end tokens), padded on
the right to `max_length` with `pad_token_id`, the end-of-text token of the CLIP
tokenizer. Returns the token indices `(max_length, num_prompts)`, 1-based as the model takes
them, and their `attention_mask`.

```julia
using HuggingFaceTokenizers
tokenizer = from_pretrained(Tokenizer, "unsloth/sam3.1")
input_ids, attention_mask = sam3_text_inputs([encode(tokenizer, "a cat").ids])
```
"""
function sam3_text_inputs(token_ids; max_length = 32, pad_token_id = 49407)
    input_ids = fill(pad_token_id + 1, max_length, length(token_ids))
    attention_mask = zeros(Int, max_length, length(token_ids))
    for (j, ids) in enumerate(token_ids)
        length(ids) <= max_length ||
            error("prompt $j has $(length(ids)) tokens, more than `max_length = $max_length`")
        input_ids[1:length(ids), j] .= ids .+ 1
        attention_mask[1:length(ids), j] .= 1
    end
    return input_ids, attention_mask
end

const Sam3Detector = Union{Sam3Model, Sam3VideoModel}

# The scores (num_queries, batch) of the queries: the probability of each times that of the
# presence of the prompt.
function sam3_scores(outputs)
    scores = sigmoid.(Array(outputs.pred_logits))
    outputs.presence_logits === nothing && return scores
    return scores .* sigmoid.(Array(outputs.presence_logits))
end

function check_target_sizes(target_sizes, b)
    target_sizes === nothing || length(target_sizes) == b ||
        error("got $(length(target_sizes)) target sizes for $b images")
end

scale_boxes(boxes, ::Nothing) = boxes
scale_boxes(boxes, (h, w)) = boxes .* eltype(boxes).([w, h, w, h])

# Probabilities (W, H, n) resized to `(height, width)`, bilinearly, as torch's
# `F.interpolate(mode = "bilinear", align_corners = False)`.
resize_probabilities(p, ::Nothing) = p
function resize_probabilities(p, (h, w))
    isempty(p) && return similar(p, w, h, 0)
    x = NNlib.upsample_bilinear(reshape(p, size(p, 1), size(p, 2), 1, :); size = (w, h),
        align_corners = false)
    return reshape(x, w, h, :)
end

"""
    post_process_object_detection(model, outputs; threshold = 0.3, target_sizes = nothing)

The objects that SAM 3 detects in each image, from the `outputs` of `model` (a
[`Sam3Model`](@ref Metalhead.Transformers.Sam3Model)), as transformers' processor of SAM 3
gives them: for each image, a NamedTuple of the `scores` of the queries above `threshold`
(the probability of each query times that of the presence of the prompt) and of their
`boxes` `(4, num_objects)`, `(x1, y1, x2, y2)` normalized to the image, or in its pixels
with the `target_sizes` `(height, width)` of the images.
"""
function post_process_object_detection(::Sam3Detector, outputs; threshold = 0.3,
        target_sizes = nothing)
    scores, boxes = sam3_scores(outputs), Array(outputs.pred_boxes)
    check_target_sizes(target_sizes, size(scores, 2))
    return map(axes(scores, 2)) do i
        keep = scores[:, i] .> threshold
        target = target_sizes === nothing ? nothing : target_sizes[i]
        return (; scores = scores[keep, i], boxes = scale_boxes(boxes[:, keep, i], target))
    end
end

"""
    post_process_instance_segmentation(model, outputs; threshold = 0.3,
                                       mask_threshold = 0.5, target_sizes = nothing)

The objects that SAM 3 segments in each image, from the `outputs` of `model` (a
[`Sam3Model`](@ref Metalhead.Transformers.Sam3Model)), as transformers' processor of SAM 3
gives them: for each image, the `scores` and `boxes` of
[`post_process_object_detection`](@ref
Metalhead.Transformers.post_process_object_detection), and the `masks` of the objects, a
`BitArray` of size `(width, height, num_objects)` where the probability of the mask is above
`mask_threshold`. With the `target_sizes` `(height, width)` of the images, the probabilities
are resized bilinearly to them first.
"""
function post_process_instance_segmentation(model::Sam3Detector, outputs; threshold = 0.3,
        mask_threshold = 0.5, target_sizes = nothing)
    detections = post_process_object_detection(model, outputs; threshold, target_sizes)
    scores = sam3_scores(outputs)
    masks = sigmoid.(Array(outputs.pred_masks))
    return map(enumerate(detections)) do (i, d)
        keep = scores[:, i] .> threshold
        target = target_sizes === nothing ? nothing : target_sizes[i]
        probabilities = resize_probabilities(masks[:, :, keep, i], target)
        return (; d.scores, d.boxes, masks = probabilities .> mask_threshold)
    end
end

"""
    post_process_semantic_segmentation(model, outputs; threshold = 0.5,
                                       target_sizes = nothing)

The semantic segmentation of each image by SAM 3, from the `outputs` of `model` (a
[`Sam3Model`](@ref Metalhead.Transformers.Sam3Model)), as transformers' processor of SAM 3
gives it: for each image, a NamedTuple of the `segmentation`, a `BitMatrix` of size
`(width, height)` where the probability of `semantic_seg` is above `threshold`, and of the
probabilities `scores`. With the `target_sizes` `(height, width)` of the images, the
probabilities are resized bilinearly to them first.
"""
function post_process_semantic_segmentation(::Sam3Detector, outputs; threshold = 0.5,
        target_sizes = nothing)
    probabilities = sigmoid.(Array(outputs.semantic_seg))
    check_target_sizes(target_sizes, size(probabilities, 4))
    return map(axes(probabilities, 4)) do i
        target = target_sizes === nothing ? nothing : target_sizes[i]
        scores = resize_probabilities(probabilities[:, :, :, i], target)[:, :, 1]
        return (; segmentation = scores .> threshold, scores)
    end
end
