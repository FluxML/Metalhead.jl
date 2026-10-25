# Ported from transformers.models.qwen2_vl.image_processing_qwen2_vl (transformers 5.20,
# Apache-2.0, The Qwen team, Alibaba Group and The Hugging Face Team):
# `Qwen2VLImageProcessor` with its torchvision backend, which Qwen3.5 checkpoints name
# `Qwen2VLImageProcessorFast`, for images (not videos). Validated against
# test/references/processors/qwen3_vl.safetensors.

# transformers' `smart_resize`: the size `(height, width)` that an image is resized to, with
# sides divisible by `factor`, an area between `min_pixels` and `max_pixels`, and the aspect
# ratio of the image as nearly as possible.
function smart_resize(height::Integer, width::Integer; factor::Integer = 28,
        min_pixels::Integer = 56 * 56, max_pixels::Integer = 14 * 14 * 4 * 1280)
    ratio = max(height, width) / min(height, width)
    ratio > 200 && error("absolute aspect ratio must be smaller than 200, got $ratio")
    h_bar = round(Int, height / factor) * factor
    w_bar = round(Int, width / factor) * factor
    if h_bar * w_bar > max_pixels
        beta = sqrt(height * width / max_pixels)
        h_bar = max(factor, floor(Int, height / beta / factor) * factor)
        w_bar = max(factor, floor(Int, width / beta / factor) * factor)
    elseif h_bar * w_bar < min_pixels
        beta = sqrt(min_pixels / (height * width))
        h_bar = ceil(Int, height * beta / factor) * factor
        w_bar = ceil(Int, width * beta / factor) * factor
    end
    return h_bar, w_bar
end

# transformers' `patchify` of one normalized image `(width, height, channels)`: the patches
# merge window by merge window, each repeated `temporal_patch_size` times, as a matrix
# `(patch_size^2 * temporal_patch_size * channels, num_patches)`.
function patchify(x, patch_size, temporal_patch_size, merge_size)
    p, m = patch_size, merge_size
    w, h, c = size(x)
    windows = reshape(x, p, m, w ÷ (p * m), p, m, h ÷ (p * m), c)
    patches = permutedims(windows, (1, 4, 7, 2, 5, 3, 6))
    patches = repeat(reshape(patches, p, p, 1, c, :), 1, 1, temporal_patch_size, 1, 1)
    return reshape(patches, p * p * temporal_patch_size * c, :)
end

"""
    Qwen2VLImageProcessor(; do_resize = true, resample = 3,
                          size = (; shortest_edge = 3136, longest_edge = 1003520),
                          do_rescale = true, rescale_factor = 1 / 255, do_normalize = true,
                          image_mean = (0.48145466, 0.4578275, 0.40821073),
                          image_std = (0.26862954, 0.26130258, 0.27577711),
                          do_convert_rgb = true, patch_size = 14, temporal_patch_size = 2,
                          merge_size = 2)

transformers' image processor of Qwen2-VL, with the defaults of its Python class; the image
processor of a [`Qwen3VLProcessor`](@ref Metalhead.Transformers.Qwen3VLProcessor), with the
settings of a repository such as `"Qwen/Qwen3.5-0.8B"` (patches of 16 pixels, mean and
standard deviation 0.5). The keys `min_pixels` and `max_pixels` of Qwen2-VL's
configurations, or of [`create_processor`](@ref Metalhead.create_processor), set
`size.shortest_edge` and `size.longest_edge`, as in transformers. Called on one image or a
vector of images, it returns the inputs of the vision encoder of Qwen3.5
([`qwen3_5_vision_model`](@ref Metalhead.Transformers.qwen3_5_vision_model)), which takes
them as they are (`model(ip(images))`), as a NamedTuple in Julia's layout (PyTorch's
dimensions reversed):

| Key              | Size               | Contents                                                                    |
|:---------------- |:------------------ |:--------------------------------------------------------------------------- |
| `pixel_values`   | `(D, num_patches)` | the patches of the images, resized and normalised, `Float32`                |
| `image_grid_thw` | `(3, num_images)`  | the numbers of patches `(1, height ÷ patch_size, width ÷ patch_size)` of each image |

where `D = 3 * temporal_patch_size * patch_size^2` (1536 for Qwen3.5). The columns of
`pixel_values` (the rows of transformers') are the patches of the images one
after the other, each image ordered by windows of `merge_size × merge_size` patches,
row-major over the windows and within each window. A patch is flattened as an array of size
`(patch_size, patch_size, temporal_patch_size, 3)` (width first), its pixels repeated
`temporal_patch_size` times: an image is a video of one frame.

An image is an array of size `(width, height, channels)` (torch's
`(channels, height, width)` reversed) of integers from 0 to 255 (an 8-bit image, as `UInt8`)
or of reals from 0 to 1. Each image is

 1. converted to RGB with `do_convert_rgb`: a grey image (a matrix, or one channel, or two
    with alpha) has its grey repeated, and the alpha of RGBA is dropped, as PIL does;
 2. resized with `do_resize`, with PIL's filter `resample`, 2 (bilinear) or 3 (bicubic),
    antialiased as torchvision resizes it, to the size whose sides are multiples of
    `patch_size * merge_size`, with an area from `size.shortest_edge` to `size.longest_edge`
    pixels, that keeps the aspect ratio of the image best (transformers' `smart_resize`).
    Without `do_resize`, the sides of an image must be multiples of
    `patch_size * merge_size`;
 3. multiplied by `rescale_factor` with `do_rescale`, and normalized with `image_mean` and
    `image_std` with `do_normalize`. An image of reals from 0 to 1 is taken as an 8-bit
    image already rescaled: `do_rescale` leaves it as it is, and without it the values are
    multiplied by 255;
 4. cut into patches of `patch_size × patch_size` pixels.

An 8-bit image gives the same `pixel_values` as transformers, and an image of reals the
same to Float32 precision. The settings are checked as the processor is made, and also
take the forms that transformers takes: `resample` as the name of PIL's filter (`:bicubic`);
`image_mean` and `image_std` as one value for all channels.
"""
Base.@kwdef struct Qwen2VLImageProcessor <: ImageProcessor
    do_resize::Bool = true
    resample::Int = 3
    size::Any = (; shortest_edge = 56 * 56, longest_edge = 28 * 28 * 1280)
    do_rescale::Bool = true
    rescale_factor::Float64 = 1 / 255
    do_normalize::Bool = true
    image_mean::Any = (0.48145466, 0.4578275, 0.40821073)
    image_std::Any = (0.26862954, 0.26130258, 0.27577711)
    do_convert_rgb::Bool = true
    patch_size::Int = 14
    temporal_patch_size::Int = 2
    merge_size::Int = 2
    provenance::NamedTuple = (;)
    function Qwen2VLImageProcessor(args...)
        ip = new(image_processor_fields(Qwen2VLImageProcessor, args)...)
        check_processor(ip)
        return ip
    end
end

register_processor!("Qwen2VLImageProcessor", Qwen2VLImageProcessor)

function ignored_keys(::Type{Qwen2VLImageProcessor})
    unread = "not read by Qwen2VLImageProcessor"
    return (; IMAGES_KWARGS_IGNORED..., do_center_crop = unread, crop_size = unread,
        do_pad = unread, pad_size = unread, image_seq_length = unread)
end
function legacy_keys(::Type{Qwen2VLImageProcessor})
    return (; min_pixels = (:size, :shortest_edge), max_pixels = (:size, :longest_edge))
end

function pixel_bounds(size)
    size isa NamedTuple && issetequal(keys(size), (:shortest_edge, :longest_edge)) &&
        all(v -> v isa Real && isinteger(v) && v > 0, values(size)) ||
        error("`size` must have positive `shortest_edge` and `longest_edge` only, as integers, got $(repr(size))")
    return Int(size.shortest_edge), Int(size.longest_edge)
end

function check_processor(ip::Qwen2VLImageProcessor)
    check_resample(ip.resample)
    pixel_bounds(ip.size)
    check_channel_stats(ip)
    for k in (:patch_size, :temporal_patch_size, :merge_size)
        getfield(ip, k) > 0 || error("`$k` must be positive, got $(getfield(ip, k))")
    end
    return nothing
end

function qwen2_vl_patches(ip::Qwen2VLImageProcessor, image)
    factor = ip.patch_size * ip.merge_size
    width, height = size(image, 1), size(image, 2)
    ip.do_resize || width % factor == 0 && height % factor == 0 ||
        error("without `do_resize`, the sides of an image must be multiples of `patch_size * merge_size` ($factor), got $width × $height")
    min_pixels, max_pixels = pixel_bounds(ip.size)
    x = prepare_image(ip, image,
        (h, w) -> smart_resize(h, w; factor, min_pixels, max_pixels))
    grid = [1, size(x, 2) ÷ ip.patch_size, size(x, 1) ÷ ip.patch_size]
    return patchify(x, ip.patch_size, ip.temporal_patch_size, ip.merge_size), grid
end

function (ip::Qwen2VLImageProcessor)(images)
    images = image_batch(images)
    isempty(images) && error("no images to process")
    processed = [qwen2_vl_patches(ip, image) for image in images]
    return (; pixel_values = reduce(hcat, first.(processed)),
        image_grid_thw = stack(last.(processed)))
end
