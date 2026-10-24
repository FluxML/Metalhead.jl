# Ported from transformers.models.qwen2_vl.image_processing_qwen2_vl (transformers 5.20,
# Apache-2.0, The Hugging Face Team): `Qwen2VLImageProcessor`, which Qwen3.5 checkpoints
# name `Qwen2VLImageProcessorFast`, for images (not videos). Validated against
# test/references/qwen2_vl_image_processor.safetensors.

"""
    smart_resize(height, width; factor = 28, min_pixels = 56 * 56, max_pixels = 14 * 14 * 4 * 1280)

transformers' `smart_resize`: the size `(height, width)` that an image is resized to, with
sides divisible by `factor`, an area between `min_pixels` and `max_pixels`, and the aspect
ratio of the image as nearly as possible.
"""
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

# The image `(width, height, channels)` resized to `(w, h)` as torchvision does on the CPU:
# antialiased bicubic interpolation in Float32. An 8-bit image is resized as PIL does, along
# its width and then along its height, rounded and clamped to 0:255 after each pass.
function resize_image(image, w, h)
    x = Float32.(image)
    bicubic(x, wh) = dropdims(
        NNlib.upsample_bicubic(reshape(x, size(x)..., 1); size = wh, antialias = true);
        dims = 4)
    eltype(image) <: Integer || return size(x)[1:2] == (w, h) ? x : bicubic(x, (w, h))
    quantize(y) = round.(clamp.(y, 0, 255))
    size(x, 1) == w || (x = quantize(bicubic(x, (w, size(x, 2)))))
    size(x, 2) == h || (x = quantize(bicubic(x, (w, h))))
    return x
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
    qwen3_5_image_inputs(images; patch_size = 16, temporal_patch_size = 2, merge_size = 2,
                         min_pixels = 65536, max_pixels = 16777216,
                         image_mean = (0.5, 0.5, 0.5), image_std = (0.5, 0.5, 0.5))
        -> (pixel_values, grid_thw)

The inputs of a Qwen3.5 vision encoder ([`qwen3_5_vision_model`](@ref
Metalhead.Transformers.qwen3_5_vision_model)) for `images`, as transformers' image
processor of Qwen3.5 (`Qwen2VLImageProcessorFast`) makes them. The defaults are those of the
`preprocessor_config.json` of the Qwen3.5 checkpoints (`min_pixels` and `max_pixels` are its
`size["shortest_edge"]` and `size["longest_edge"]`).

`images` is one image or a vector of images. An image is an array of size
`(width, height, channels)` (torch's `(channels, height, width)` reversed) of integers from
0 to 255 (an 8-bit image, as `UInt8`) or of real numbers from 0 to 1. For an Images.jl
image `img`, that is `permutedims(channelview(img), (3, 2, 1))`. Each image is

 1. resized with antialiased bicubic interpolation to the [`smart_resize`](@ref
    Metalhead.Transformers.smart_resize) of its size with `factor = patch_size * merge_size`
    (an 8-bit image along its width and then its height, rounded back to integers after
    each, as torchvision does on the CPU and PIL),
 2. scaled to `[0, 1]` (8-bit images only) and normalized with `image_mean` and `image_std`,
 3. cut into patches of `patch_size × patch_size` pixels, each repeated
    `temporal_patch_size` times (an image is a video of one frame).

Returns `pixel_values`, a `Float32` matrix of size
`(patch_size^2 * temporal_patch_size * channels, num_patches)` whose columns are the patches
of the images one after the other, each image ordered by windows of `merge_size` by
`merge_size` patches, and `grid_thw`, an integer matrix of size `(3, num_images)` whose
columns are the numbers of patches `(1, height ÷ patch_size, width ÷ patch_size)` of the
images.

An 8-bit image whose size needs no resizing gives the same `pixel_values` as transformers,
and an image of real numbers the same to Float32 precision. A resized 8-bit image may differ
by one level of 255 in about 1% of its values, as torchvision computes its weights in
fixed-point arithmetic.
"""
function qwen3_5_image_inputs(images; patch_size::Integer = 16,
        temporal_patch_size::Integer = 2, merge_size::Integer = 2,
        min_pixels::Integer = 65536, max_pixels::Integer = 16777216,
        image_mean = (0.5, 0.5, 0.5), image_std = (0.5, 0.5, 0.5))
    images isa AbstractArray{<:Real, 3} && return qwen3_5_image_inputs([images];
        patch_size, temporal_patch_size, merge_size, min_pixels, max_pixels, image_mean,
        image_std)
    patches = map(images) do image
        ndims(image) == 3 ||
            error("an image must be an array of size (width, height, channels), got $(size(image))")
        width, height, channels = size(image)
        channels == length(image_mean) == length(image_std) ||
            error("an image has $channels channels, `image_mean` and `image_std` $(length(image_mean)) and $(length(image_std))")
        h, w = smart_resize(height, width; factor = patch_size * merge_size, min_pixels,
            max_pixels)
        # transformers folds the rescaling of 8-bit images into the mean and the std
        scale = eltype(image) <: Integer ? 255.0f0 : 1.0f0
        mean = reshape(Float32.(collect(image_mean)) .* scale, 1, 1, :)
        std = reshape(Float32.(collect(image_std)) .* scale, 1, 1, :)
        x = (resize_image(image, w, h) .- mean) ./ std
        grid = [1, h ÷ patch_size, w ÷ patch_size]
        return patchify(x, patch_size, temporal_patch_size, merge_size), grid
    end
    return reduce(hcat, first.(patches)), reduce(hcat, last.(patches))
end
