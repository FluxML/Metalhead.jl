# Ported from transformers.models.clip.image_processing_clip (transformers 5.20,
# Apache-2.0, The Hugging Face Team): `CLIPImageProcessor`, with its torchvision backend,
# which CLIP checkpoints name `CLIPFeatureExtractor` or `CLIPImageProcessor`. Validated
# against test/references/processors/clip.safetensors.

const OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
const OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

"""
    CLIPImageProcessor(; do_resize = true, size = (; shortest_edge = 224), resample = 3,
                       do_center_crop = true, crop_size = (; height = 224, width = 224),
                       do_rescale = true, rescale_factor = 1 / 255, do_normalize = true,
                       image_mean = (0.48145466, 0.4578275, 0.40821073),
                       image_std = (0.26862954, 0.26130258, 0.27577711),
                       do_convert_rgb = true, default_to_square = false, do_pad = false)

transformers' image processor of CLIP, with the defaults of its Python class; the image
processor of a [`CLIPProcessor`](@ref Metalhead.Transformers.CLIPProcessor), with the
settings of a repository such as `"openai/clip-vit-base-patch32"`. Called on one image or a
vector of images, it returns the input of the vision encoder of CLIP, which takes it as it
is (`model(ip(images))`), as a NamedTuple:

| Key            | Size               | Contents                                               |
|:-------------- |:------------------ |:------------------------------------------------------ |
| `pixel_values` | `(W, H, 3, batch)` | the images, resized, cropped and normalised, `Float32` |

An image is an array of size `(width, height, channels)` (torch's
`(channels, height, width)` reversed) of integers from 0 to 255 (an 8-bit image, as `UInt8`)
or of reals from 0 to 1. Each image is

 1. converted to RGB with `do_convert_rgb`: a grey image (a matrix, or one channel, or two
    with alpha) has its grey repeated, and the alpha of RGBA is dropped, as PIL does;
 2. resized with `do_resize`, with PIL's filter `resample`, 2 (bilinear) or 3 (bicubic),
    antialiased as torchvision resizes it: its shorter side to `size.shortest_edge` and the
    longer alike, rounded down, or to `size.height` and `size.width`;
 3. center-cropped to `crop_size.height` and `crop_size.width` with `do_center_crop`, from
    offsets rounded down, and padded with zeros where it is smaller;
 4. multiplied by `rescale_factor` with `do_rescale`, and normalized with `image_mean` and
    `image_std` with `do_normalize`. An image of reals from 0 to 1 is taken as an 8-bit
    image already rescaled: `do_rescale` leaves it as it is, and without it the values are
    multiplied by 255.

An 8-bit image gives the same `pixel_values` as transformers, and an image of reals the
same to Float32 precision. The settings are checked as the processor is made, and also
take the forms that transformers takes: `size` as an integer, the shorter side (or, with
`default_to_square`, the side of a square); `crop_size` as an integer, the side of a
square; `resample` as the name of PIL's filter (`:bicubic`); `image_mean` and `image_std`
as one value for all channels. The padding of transformers' image processors isn't ported:
`do_pad` is `false` (or `nothing`, Python's `None`).
"""
@processor struct CLIPImageProcessor <: AbstractImageProcessor
    do_resize::Bool = true
    size::Any = (; shortest_edge = 224)
    resample::Int = 3
    do_center_crop::Bool = true
    crop_size::Any = (; height = 224, width = 224)
    do_rescale::Bool = true
    rescale_factor::Float64 = 1 / 255
    do_normalize::Bool = true
    image_mean::Any = OPENAI_CLIP_MEAN
    image_std::Any = OPENAI_CLIP_STD
    do_convert_rgb::Bool = true
    default_to_square::Bool = false
    do_pad::Bool = false
    provenance::NamedTuple = (;)
end

register_processor!("CLIPImageProcessor", CLIPImageProcessor;
    aliases = ["CLIPFeatureExtractor"])

unported_image_features(::Type{CLIPImageProcessor}) = (:do_pad,)

function standardize_settings(T::Type{CLIPImageProcessor}, settings::NamedTuple)
    s = standardize_image_settings(T, settings)
    # transformers' `get_size_dict` of integers, after the other settings are read
    size = s.size isa Integer && !(s.size isa Bool) ?
           (s.default_to_square ? (; height = s.size, width = s.size) :
            (; shortest_edge = s.size)) : s.size
    crop_size = s.crop_size isa Integer && !(s.crop_size isa Bool) ?
                (; height = s.crop_size, width = s.crop_size) : s.crop_size
    s = merge(s, (; size, crop_size))
    clip_resize_size(s.size)
    clip_crop_size(s.crop_size)
    return s
end

function ignored_keys(::Type{CLIPImageProcessor})
    return (; Base.structdiff(IMAGES_KWARGS_IGNORED, NamedTuple{(:default_to_square,)})...,
        pad_size = "only used with `do_pad`")
end

function clip_positive_integers(v)
    return all(x -> x isa Real && !(x isa Bool) && isinteger(x) && x > 0, v)
end

# The `size` of the resize, as `(; shortest_edge)` or `(height, width)`: the forms of
# the torchvision backend's `resize` that CLIP checkpoints use.
function clip_resize_size(size)
    if size isa NamedTuple && clip_positive_integers(values(size))
        keys(size) == (:shortest_edge,) && return Int(size.shortest_edge)
        issetequal(keys(size), (:height, :width)) &&
            return (Int(size.height), Int(size.width))
    end
    return throw(ArgumentError("`size` must have `shortest_edge` only, or `height` and `width` only, as positive integers, got $(repr(size))"))
end

function clip_crop_size(crop_size)
    crop_size isa NamedTuple && issetequal(keys(crop_size), (:height, :width)) &&
        clip_positive_integers(values(crop_size)) ||
        throw(ArgumentError("`crop_size` must have `height` and `width` only, as positive integers, got $(repr(crop_size))"))
    return Int(crop_size.height), Int(crop_size.width)
end

function (ip::CLIPImageProcessor)(images)
    images = image_batch(images)
    isempty(images) && error("no images to process")
    size = clip_resize_size(ip.size)
    target_size = size isa Integer ? (h, w) -> get_resize_output_image_size(h, w, size) :
                  Returns(size)
    crop_size = ip.do_center_crop ? clip_crop_size(ip.crop_size) : nothing
    pixel_values = stack_images([prepare_image(ip, image, target_size; crop_size)
                                 for image in images])
    return (; pixel_values)
end
