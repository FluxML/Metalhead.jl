# The images that processors take, in one layout: every image goes through `image_array`, so
# that a package extension supporting another type of image adds a method to it.

"""
    image_array(image) -> AbstractArray{<:Real, 3}

`image` as processors take it: an array of size `(width, height, channels)` (torch's
`(channels, height, width)` reversed) of integers from 0 to 255 (an 8-bit image, such as
`UInt8`) or of reals from 0 to 1. A matrix is a grey image of one channel.

With ImageCore loaded (as by Images.jl), it also takes a matrix of colorants of size
`(height, width)`, such as the `Matrix{RGB{N0f8}}` that `FileIO.load` returns, as
`permutedims(channelview(img), (3, 2, 1))` but with the channels of any colorant in the
order R, G, B (then alpha), or gray (then alpha). `N0f8` values, those of 8-bit images,
become the `UInt8` values they store, and others reals from 0 to 1.
"""
image_array(x::AbstractArray{<:Real, 3}) = x
image_array(x::AbstractMatrix{<:Real}) = reshape(x, size(x)..., 1)
function image_array(x)
    throw(ArgumentError("an image must be an array of size (width, height, channels) of integers from 0 to 255 or of reals from 0 to 1, got a $(typeof(x))"))
end

# The images of `images`, one image or a vector of them (or an array of size
# `(width, height, channels, batch)`), each through `image_array`.
image_batch(images::AbstractVector) = [image_array(x) for x in images]
function image_batch(images::AbstractArray{<:Real, 4})
    return [image_array(x) for x in eachslice(images; dims = 4)]
end
image_batch(image) = [image_array(image)]

# Whether an image holds 8-bit values (0 to 255), rather than reals from 0 to 1.
is_8bit(image) = eltype(image) <: Integer

# PIL's `convert("RGB")`, which transformers applies with `do_convert_rgb`: the grey of an
# image of one channel (or of grey and alpha) is repeated, and the alpha of RGBA is dropped.
function convert_to_rgb(x::AbstractArray{<:Real, 3})
    c = size(x, 3)
    c == 3 && return x
    c in (1, 2) && return repeat(x[:, :, 1:1], 1, 1, 3)
    c == 4 && return x[:, :, 1:3]
    throw(ArgumentError("an image has 1 to 4 channels (L, LA, RGB or RGBA), got $c"))
end

# The resampling filters of PIL, by the integer of its enumeration that configurations
# store as `resample`, named as `Timm.pil_resize` names them.
const RESAMPLE_FILTERS = Dict(0 => :nearest, 1 => :lanczos, 2 => :bilinear, 3 => :bicubic,
    4 => :box, 5 => :hamming)
