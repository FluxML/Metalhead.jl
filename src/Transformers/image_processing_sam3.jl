# Ported from transformers.models.sam3.image_processing_sam3 (transformers 5.20, Apache-2.0,
# The Hugging Face Team): `Sam3ImageProcessor`, of images (not segmentation maps), and its
# post-processing of the outputs of SAM 3. Validated against
# test/references/processors/sam3.safetensors.

"""
    Sam3ImageProcessor(; do_resize = true, size = (; height = 1008, width = 1008),
                       resample = 2, do_rescale = true, rescale_factor = 1 / 255,
                       do_normalize = true, image_mean = (0.5, 0.5, 0.5),
                       image_std = (0.5, 0.5, 0.5), do_convert_rgb = true)

transformers' image processor of SAM 3, with the defaults of its Python class; the image
processor of a [`Sam3Processor`](@ref Metalhead.Transformers.Sam3Processor), with the
settings of a repository. Called on one image or a vector of images, it returns a
NamedTuple:

| Key              | Size               | Contents                                                                   |
|:---------------- |:------------------ |:-------------------------------------------------------------------------- |
| `pixel_values`   | `(W, H, 3, batch)` | the images, resized and normalised, `Float32`                              |
| `original_sizes` | `batch`            | a tuple of the `(height, width)` of each image, for the boxes and post-processing; `gpu` leaves it on the CPU |

An image is an array of size `(width, height, channels)` (torch's
`(channels, height, width)` reversed) of integers from 0 to 255 (an 8-bit image, as `UInt8`)
or of reals from 0 to 1. Each image is

 1. converted to RGB with `do_convert_rgb`: a grey image (a matrix, or one channel, or two
    with alpha) has its grey repeated, and the alpha of RGBA is dropped, as PIL does;
 2. resized to `size` with `do_resize`, with PIL's filter `resample`, 2 (bilinear) or 3
    (bicubic), antialiased as torchvision resizes it;
 3. multiplied by `rescale_factor` with `do_rescale`, and normalized with `image_mean` and
    `image_std` with `do_normalize`. An image of reals from 0 to 1 is taken as an 8-bit
    image already rescaled: `do_rescale` leaves it as it is, and without it the values are
    multiplied by 255.

An 8-bit image gives the same `pixel_values` as transformers, and an image of reals the
same to Float32 precision. The settings are checked as the processor is made, and also
take the forms that transformers takes: `size` as an integer, the side of a square;
`resample` as the name of PIL's filter (`:bilinear`); `image_mean` and `image_std` as one
value for all channels.
"""
@processor struct Sam3ImageProcessor <: ImageProcessor
    do_resize::Bool = true
    size::Any = (; height = 1008, width = 1008)
    resample::Int = 2
    do_rescale::Bool = true
    rescale_factor::Float64 = 1 / 255
    do_normalize::Bool = true
    image_mean::Any = (0.5, 0.5, 0.5)
    image_std::Any = (0.5, 0.5, 0.5)
    do_convert_rgb::Bool = true
    provenance::NamedTuple = (;)
end

register_processor!("Sam3ImageProcessor", Sam3ImageProcessor)

processor_fields(T::Type{Sam3ImageProcessor}, args::Tuple) = image_processor_fields(T, args)

# transformers' `get_size_dict(size, default_to_square = True)` of an integer
size_setting(::Type{Sam3ImageProcessor}, v::Integer) = (; height = v, width = v)

function ignored_keys(::Type{Sam3ImageProcessor})
    return (; IMAGES_KWARGS_IGNORED...,
        mask_size = "only resizes the segmentation maps of training labels",
        crop_size = "only used with `do_center_crop`",
        pad_size = "only used with `do_pad`",
        image_seq_length = "not read by Sam3ImageProcessor")
end
function restricted_keys(::Type{Sam3ImageProcessor})
    return (; do_center_crop = (false,), do_pad = (false,))
end

function check_processor(ip::Sam3ImageProcessor)
    check_resample(ip.resample)
    size_hw(ip.size)
    check_channel_stats(ip)
    return nothing
end

function (ip::Sam3ImageProcessor)(images)
    images = image_batch(images)
    isempty(images) && error("no images to process")
    target_size = Returns(size_hw(ip.size))
    pixel_values = stack_images([prepare_image(ip, image, target_size) for image in images])
    return (; pixel_values, original_sizes = Tuple((size(x, 2), size(x, 1)) for x in images))
end

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

# torch's `F.interpolate(mode = "bilinear", align_corners = False)`
resize_probabilities(p, ::Nothing) = p
function resize_probabilities(p, (h, w))
    isempty(p) && return similar(p, w, h, 0)
    x = NNlib.upsample_bilinear(reshape(p, size(p, 1), size(p, 2), 1, :); size = (w, h),
        align_corners = false)
    return reshape(x, w, h, :)
end

"""
    post_process_object_detection(processor, outputs; threshold = 0.3,
                                  target_sizes = nothing)

The objects that SAM 3 detects in each image, from the `outputs` of a [`Sam3Model`](@ref
Metalhead.Transformers.Sam3Model), as transformers' processor of SAM 3 gives them:
`processor` is a [`Sam3Processor`](@ref Metalhead.Transformers.Sam3Processor) or its image
processor. For each image, a NamedTuple with the keys of transformers' result: the
`scores` of the queries above `threshold` (the probability of each query times that of the
presence of the prompt) and their `boxes` `(4, num_objects)`, `(x1, y1, x2, y2)` normalized
to the image, or in its pixels with the `target_sizes` `(height, width)` of the images (such
as the `original_sizes` of the processor's output).
"""
function post_process_object_detection(::Sam3ImageProcessor, outputs; threshold = 0.3,
        target_sizes = nothing)
    target_sizes = size_list(target_sizes)
    scores, boxes = sam3_scores(outputs), Array(outputs.pred_boxes)
    check_target_sizes(target_sizes, size(scores, 2))
    return map(axes(scores, 2)) do i
        keep = scores[:, i] .> threshold
        target = target_sizes === nothing ? nothing : target_sizes[i]
        return (; scores = scores[keep, i], boxes = scale_boxes(boxes[:, keep, i], target))
    end
end

"""
    post_process_instance_segmentation(processor, outputs; threshold = 0.3,
                                       mask_threshold = 0.5, target_sizes = nothing)

The objects that SAM 3 segments in each image, from the `outputs` of a [`Sam3Model`](@ref
Metalhead.Transformers.Sam3Model), as transformers' processor of SAM 3 gives them:
`processor` is a [`Sam3Processor`](@ref Metalhead.Transformers.Sam3Processor) or its image
processor. For each image, a NamedTuple with the keys of transformers' result: the `scores`
and `boxes` of [`post_process_object_detection`](@ref
Metalhead.Transformers.post_process_object_detection), and the `masks` of the objects, a
`BitArray` of size `(width, height, num_objects)` where the probability of the mask is above
`mask_threshold`. With the `target_sizes` `(height, width)` of the images, the probabilities
are resized bilinearly to them first.
"""
function post_process_instance_segmentation(ip::Sam3ImageProcessor, outputs;
        threshold = 0.3, mask_threshold = 0.5, target_sizes = nothing)
    target_sizes = size_list(target_sizes)
    detections = post_process_object_detection(ip, outputs; threshold, target_sizes)
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
    post_process_semantic_segmentation(processor, outputs; threshold = 0.5,
                                       target_sizes = nothing)

The semantic segmentation of each image by SAM 3, from the `outputs` of a
[`Sam3Model`](@ref Metalhead.Transformers.Sam3Model), as transformers' processor of SAM 3
gives it: `processor` is a [`Sam3Processor`](@ref Metalhead.Transformers.Sam3Processor) or
its image processor. For each image, a NamedTuple with the keys of transformers' result
with `return_segmentation_scores = True`: the `segmentation`, a `BitMatrix` of size
`(width, height)` where the probability of `semantic_seg` is above `threshold`, and the
probabilities `segmentation_scores`, of size `(width, height, 1)`. With the `target_sizes`
`(height, width)` of the images, the probabilities are resized bilinearly to them first.
"""
function post_process_semantic_segmentation(::Sam3ImageProcessor, outputs; threshold = 0.5,
        target_sizes = nothing)
    target_sizes = size_list(target_sizes)
    probabilities = sigmoid.(Array(outputs.semantic_seg))
    check_target_sizes(target_sizes, size(probabilities, 4))
    return map(axes(probabilities, 4)) do i
        target = target_sizes === nothing ? nothing : target_sizes[i]
        segmentation_scores = resize_probabilities(probabilities[:, :, :, i], target)
        return (; segmentation = segmentation_scores[:, :, 1] .> threshold,
            segmentation_scores)
    end
end
