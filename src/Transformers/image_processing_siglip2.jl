# Ported from transformers.models.siglip2.image_processing_siglip2 (transformers 5.21,
# Apache-2.0, The Hugging Face Team): `Siglip2ImageProcessor`, the image processor of the
# NaFlex checkpoints of SigLIP 2, which checkpoints name `Siglip2ImageProcessorFast`.
# Validated against test/references/processors/siglip2.safetensors.

# transformers' `get_image_size_for_max_num_patches`: the size `(height, width)`, in
# multiples of `patch_size` and of at least one patch, of the largest scale of the image
# with at most `max_num_patches` patches, found by bisection.
function image_size_for_max_num_patches(height::Integer, width::Integer,
        patch_size::Integer, max_num_patches::Integer; eps = 1.0e-5)
    scaled(scale, side) = max(patch_size, ceil(Int, side * scale / patch_size) * patch_size)
    scale_min, scale_max = eps / 10, 100.0
    while scale_max - scale_min >= eps
        scale = (scale_min + scale_max) / 2
        h, w = scaled(scale, height), scaled(scale, width)
        if (h / patch_size) * (w / patch_size) <= max_num_patches
            scale_min = scale
        else
            scale_max = scale
        end
    end
    return scaled(scale_min, height), scaled(scale_min, width)
end

"""
    Siglip2ImageProcessor(; do_resize = true, resample = 2, do_rescale = true,
                          rescale_factor = 1 / 255, do_normalize = true,
                          image_mean = (0.5, 0.5, 0.5), image_std = (0.5, 0.5, 0.5),
                          do_convert_rgb = false, patch_size = 16, max_num_patches = 256)

transformers' image processor of the NaFlex checkpoints of SigLIP 2, with the defaults of
its Python class; the image processor of a [`Siglip2Processor`](@ref
Metalhead.Transformers.Siglip2Processor), with the settings of a repository such as
`"google/siglip2-base-patch16-naflex"`. It keeps the aspect ratio of each image, which it
cuts into at most `max_num_patches` patches. Called on one image or a vector of images, it
returns the inputs of the vision model of SigLIP 2 ([`Siglip2VisionModel`](@ref
Metalhead.Transformers.Siglip2VisionModel)), which takes them as they are, as a
NamedTuple in Julia's layout (PyTorch's dimensions reversed):

| Key                    | Size                               | Contents                                                     |
|:---------------------- |:---------------------------------- |:------------------------------------------------------------ |
| `pixel_values`         | `(D, max_num_patches, batch)`      | the patches of each image, resized and normalised, `Float32`, padded with zeros |
| `pixel_attention_mask` | `(max_num_patches, batch)`         | 1 for the patches of each image, 0 for the padding           |
| `spatial_shapes`       | `(2, batch)`                       | the numbers of patches `(height, width)` of each image       |

where `D = 3 * patch_size^2` (768 for SigLIP 2). The patches of an image are row-major over
its grid, and each is flattened as an array of size `(3, patch_size, patch_size)`
(channels first, then width), as transformers' `convert_image_to_patches` flattens them.

An image is an array of size `(width, height, channels)` (torch's
`(channels, height, width)` reversed) of integers from 0 to 255 (an 8-bit image, as `UInt8`)
or of reals from 0 to 1. Each image is

 1. converted to RGB with `do_convert_rgb` (`false` by default, Python's `None`): a grey
    image (a matrix, or one channel, or two with alpha) has its grey repeated, and the
    alpha of RGBA is dropped, as PIL does;
 2. resized with `do_resize`, with PIL's filter `resample`, 2 (bilinear) or 3 (bicubic),
    antialiased as torchvision resizes it, to the largest size in multiples of
    `patch_size` with at most `max_num_patches` patches whose sides scale the image by the
    same factor, rounded up (transformers' `get_image_size_for_max_num_patches`). Without
    `do_resize`, the sides of an image must be multiples of `patch_size`;
 3. multiplied by `rescale_factor` with `do_rescale`, and normalized with `image_mean` and
    `image_std` with `do_normalize`. An image of reals from 0 to 1 is taken as an 8-bit
    image already rescaled: `do_rescale` leaves it as it is, and without it the values are
    multiplied by 255;
 4. cut into patches of `patch_size × patch_size` pixels, padded to `max_num_patches`.

An 8-bit image gives the same `pixel_values` as transformers, and an image of reals the
same to Float32 precision. The settings are checked as the processor is made, and also
take the forms that transformers takes: `resample` as the name of PIL's filter
(`:bilinear`); `image_mean` and `image_std` as one value for all channels.
"""
@processor struct Siglip2ImageProcessor <: AbstractImageProcessor
    do_resize::Bool = true
    resample::Int = 2
    do_rescale::Bool = true
    rescale_factor::Float64 = 1 / 255
    do_normalize::Bool = true
    image_mean::Any = (0.5, 0.5, 0.5)
    image_std::Any = (0.5, 0.5, 0.5)
    do_convert_rgb::Bool = false
    patch_size::Int = 16
    max_num_patches::Int = 256
    provenance::NamedTuple = (;)
end

register_processor!("Siglip2ImageProcessor", Siglip2ImageProcessor)

function standardize_settings(T::Type{Siglip2ImageProcessor}, settings::NamedTuple)
    s = standardize_image_settings(T, settings)
    for k in (:patch_size, :max_num_patches)
        s[k] > 0 || throw(ArgumentError("`$k` must be positive, got $(s[k])"))
    end
    return s
end

function ignored_keys(::Type{Siglip2ImageProcessor})
    unread = "not read by Siglip2ImageProcessor"
    return (; IMAGES_KWARGS_IGNORED...,
        size = "not read by Siglip2ImageProcessor, which resizes to `max_num_patches`",
        do_center_crop = unread, crop_size = unread, do_pad = unread, pad_size = unread,
        image_seq_length = unread)
end

# The patches (D, max_num_patches) of an image, padded with zeros, their mask and the grid
# (height, width) of the image.
function siglip2_patches(ip::Siglip2ImageProcessor, image)
    p, max_patches = ip.patch_size, ip.max_num_patches
    x = prepare_image(ip, image,
        (h, w) -> image_size_for_max_num_patches(h, w, p, max_patches))
    w, h, c = size(x)
    w % p == 0 && h % p == 0 ||
        error("without `do_resize`, the sides of an image must be multiples of `patch_size` ($p), got $w × $h")
    nw, nh = w ÷ p, h ÷ p
    n = nw * nh
    n <= max_patches ||
        error("an image of $nw × $nh patches has more than `max_num_patches = $max_patches`")
    patches = reshape(permutedims(reshape(x, p, nw, p, nh, c), (5, 1, 3, 2, 4)), :, n)
    padded = hcat(patches, zeros(Float32, size(patches, 1), max_patches - n))
    return padded, [ones(Int, n); zeros(Int, max_patches - n)], [nh, nw]
end

function (ip::Siglip2ImageProcessor)(images)
    images = image_batch(images)
    isempty(images) && error("no images to process")
    processed = [siglip2_patches(ip, image) for image in images]
    channels = unique(size(first(x), 1) for x in processed)
    length(channels) == 1 ||
        error("images of different numbers of channels can't be batched; convert them to RGB (`do_convert_rgb = true`)")
    return (; pixel_values = stack(first.(processed)),
        pixel_attention_mask = stack(getindex.(processed, 2)),
        spatial_shapes = stack(last.(processed)))
end
