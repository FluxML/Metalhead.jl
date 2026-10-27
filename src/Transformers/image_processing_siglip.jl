# Ported from transformers.models.siglip.image_processing_siglip (transformers 5.21,
# Apache-2.0, The Hugging Face Team): `SiglipImageProcessor`, the image processor of SigLIP
# and of the SigLIP 2 checkpoints of fixed resolution. Validated against
# test/references/processors/siglip.safetensors.

"""
    SiglipImageProcessor(; do_resize = true, size = (; height = 224, width = 224),
                         resample = 3, do_rescale = true, rescale_factor = 1 / 255,
                         do_normalize = true, image_mean = (0.5, 0.5, 0.5),
                         image_std = (0.5, 0.5, 0.5), do_convert_rgb = true,
                         do_center_crop = false, do_pad = false)

transformers' image processor of SigLIP, with the defaults of its Python class; the image
processor of a [`SiglipProcessor`](@ref Metalhead.Transformers.SiglipProcessor), with the
settings of a repository (the SigLIP 2 checkpoints of fixed resolution resize bilinearly,
`resample = 2`). Called on one image or a vector of images, it returns a NamedTuple:

| Key            | Size               | Contents                                       |
|:-------------- |:------------------ |:---------------------------------------------- |
| `pixel_values` | `(W, H, 3, batch)` | the images, resized and normalised, `Float32` |

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
take the forms that transformers takes: `resample` as the name of PIL's filter
(`:bicubic`); `image_mean` and `image_std` as one value for all channels. An integer `size`,
which resizes the shortest edge in transformers' SigLIP (`default_to_square = False`), the
center crop and the padding aren't ported: `do_center_crop` and `do_pad` are `false` (or
`nothing`, Python's `None`).
"""
@processor struct SiglipImageProcessor <: AbstractImageProcessor
    do_resize::Bool = true
    size::Any = (; height = 224, width = 224)
    resample::Int = 3
    do_rescale::Bool = true
    rescale_factor::Float64 = 1 / 255
    do_normalize::Bool = true
    image_mean::Any = (0.5, 0.5, 0.5)
    image_std::Any = (0.5, 0.5, 0.5)
    do_convert_rgb::Bool = true
    do_center_crop::Bool = false
    do_pad::Bool = false
    provenance::NamedTuple = (;)
end

register_processor!("SiglipImageProcessor", SiglipImageProcessor)

function standardize_settings(T::Type{SiglipImageProcessor}, settings::NamedTuple)
    s = standardize_image_settings(T, settings)
    size_hw(s.size)
    return s
end

function size_setting(::Type{SiglipImageProcessor}, v::Integer)
    throw(ArgumentError("`size = $v`: an integer `size` resizes the shortest edge in SiglipImageProcessor (`default_to_square = False`), which isn't ported; give `(; height, width)`"))
end

function ignored_keys(::Type{SiglipImageProcessor})
    return (; IMAGES_KWARGS_IGNORED...,
        crop_size = "only used with `do_center_crop`",
        pad_size = "only used with `do_pad`",
        image_seq_length = "not read by SiglipImageProcessor")
end

function (ip::SiglipImageProcessor)(images)
    images = image_batch(images)
    isempty(images) && error("no images to process")
    target_size = Returns(size_hw(ip.size))
    return (; pixel_values = stack_images([prepare_image(ip, image, target_size)
                                           for image in images]))
end
