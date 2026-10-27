# Ported from transformers.image_processing_backends and image_processing_utils
# (transformers 5.20, Apache-2.0, The Hugging Face Team): the operations of
# `TorchvisionBackend`, the default backend of transformers' image processors, on the CPU,
# shared by the image processors of the models.

# The settings of transformers' `ImagesKwargs` that no image processor of Metalhead reads,
# for the `ignored_keys` of each.
const IMAGES_KWARGS_IGNORED = (;
    data_format = "the torchvision backend returns channels first whatever it says",
    input_data_format = "Metalhead's images are (width, height, channels)",
    device = "processors run on the CPU",
    return_tensors = "processors return Julia arrays",
    disable_grouping = "an optimisation of the torchvision backend",
    default_to_square = "only converts an integer `size`, which configurations don't have")

# The filters of transformers' `resample` (PIL's enumeration) that the torchvision backend
# resizes with and are ported.
const RESAMPLE_PORTED = (2, 3)

resample_supported() = join(("$r ($(RESAMPLE_FILTERS[r]))" for r in RESAMPLE_PORTED), ", ")

function check_resample(resample)
    resample in RESAMPLE_PORTED && return nothing
    name = get(RESAMPLE_FILTERS, resample, "unknown")
    throw(ArgumentError("`resample = $(repr(resample))` ($name) is not supported (supported: $(resample_supported()))"))
end

# The features of transformers' image processors that the image processor of type `T`
# doesn't port, as settings whose only supported value is `false` (`None` in Python, the
# default of the classes that don't have them).
const UNPORTED_IMAGE_FEATURES = (:do_center_crop, :do_pad)
unported_image_features(::Type) = UNPORTED_IMAGE_FEATURES

# The `standardize_settings` of the image processors: the forms that transformers takes
# (`resample = :bilinear`, an integer `size`, one `image_mean` for every channel) in the
# types of the fields, and the settings that aren't ported rejected.
function standardize_image_settings(::Type{T}, s::NamedTuple) where {T}
    s = NamedTuple{keys(s)}(map((name, v) -> field_setting(T, name, v), keys(s), values(s)))
    for name in unported_image_features(T)
        haskey(s, name) && s[name] &&
            throw(ArgumentError("`$name = true` is not supported by $(nameof(T)) (supported: false)"))
    end
    check_resample(s.resample)
    check_channel_stats(s)
    return s
end

# `size` as the image processor of type `T` takes it; checked by its `standardize_settings`.
size_setting(T, v) = v

function field_setting(T, name, v)
    name === :size && return size_setting(T, v)
    name in unported_image_features(T) && v === nothing && return false
    name === :resample && v isa Union{Symbol, AbstractString} && return resample_number(v)
    name in (:image_mean, :image_std) && return stats_setting(name, v)
    F = fieldtype(T, name)
    v isa F && return v
    F === Int && v isa Real && !(v isa Bool) && isinteger(v) && return Int(v)
    F === Float64 && v isa Real && !(v isa Bool) && return Float64(v)
    form = F === Bool ? "true or false" :
           F === Int ? "an integer" :
           F === Float64 ? "a number" : "a $F"
    throw(ArgumentError("`$name = $(repr(v))`: `$name` of $(nameof(T)) is $form"))
end

function resample_number(name)
    for (k, filter) in RESAMPLE_FILTERS
        filter === Symbol(name) && return k
    end
    throw(ArgumentError("`resample = $(repr(name))` is not a filter of PIL (supported: $(resample_supported()))"))
end

function stats_setting(name, v)
    v isa Real && !(v isa Bool) && return Float64(v)
    v isa Union{Tuple, AbstractVector} && !isempty(v) && all(x -> x isa Real, v) &&
        return Tuple(Float64.(v))
    throw(ArgumentError("`$name = $(repr(v))`: `$name` is a number, or one per channel"))
end

# With `do_convert_rgb`, images have 3 channels, so that other statistics fail at loading.
function check_channel_stats(s)
    s.do_convert_rgb || return nothing
    for name in (:image_mean, :image_std)
        v = getfield(s, name)
        v isa Real || length(v) == 3 ||
            throw(ArgumentError("`$name` has one value per RGB channel, or one for all, got $(repr(v))"))
    end
    return nothing
end

# The image `(width, height, channels)` resized to `(w, h)` with PIL's filter `resample`
# (2, bilinear, or 3, bicubic), antialiased, as transformers' torchvision backend resizes it
# on the CPU: an 8-bit image as torchvision resizes a `uint8` tensor, in fixed point, bit
# for bit (`Timm.pil_resize`: torchvision's int16 path, which x86 CPUs with AVX2 take for
# bilinear and all CPUs for bicubic), and an image of reals in Float32 (NNlib's antialiased
# interpolation, to Float32 precision).
function resize_image(image, (w, h), resample)
    check_resample(resample)
    filter = RESAMPLE_FILTERS[resample]
    size(image)[1:2] == (w, h) && return Float32.(image)
    is_8bit(image) &&
        return Float32.(pil_resize(UInt8.(image), (w, h), filter; torchvision = true))
    upsample = filter === :bilinear ? NNlib.upsample_bilinear : NNlib.upsample_bicubic
    x = reshape(Float32.(image), size(image)..., 1)
    return dropdims(upsample(x; size = (w, h), antialias = true, align_corners = false);
        dims = 4)
end

# The `image_mean` or `image_std` of the channels of `x`, of size (1, 1, channels).
function channel_stats(x, stats, name)
    c = size(x, 3)
    v = stats isa Real ? fill(Float32(stats), c) : Float32.(collect(stats))
    length(v) == c ||
        error("an image has $c channels, `$name` $(length(v)) values")
    return reshape(v, 1, 1, c)
end

# The Float32 image `x` (width, height, channels) rescaled and normalized as transformers'
# torchvision backend does, which folds `rescale_factor` into the mean and the standard
# deviation when it does both. `eightbit` says whether `x` holds the values of an 8-bit
# image, 0 to 255, or reals from 0 to 1, which are taken as an 8-bit image already
# multiplied by 1/255: `do_rescale` with the usual `rescale_factor = 1/255` leaves them as
# they are, and without `do_rescale` they are multiplied by 255.
function rescale_and_normalize(x, eightbit, do_rescale, rescale_factor, do_normalize,
        image_mean, image_std)
    if !eightbit
        scale = do_rescale ? 255 * rescale_factor : 255.0
        isapprox(scale, 1; rtol = 1.0e-6) || (x = x .* Float32(scale))
        do_normalize || return x
        return (x .- channel_stats(x, image_mean, "image_mean")) ./
               channel_stats(x, image_std, "image_std")
    end
    if do_rescale && do_normalize
        s = Float32(1 / rescale_factor)
        return (x .- channel_stats(x, image_mean, "image_mean") .* s) ./
               (channel_stats(x, image_std, "image_std") .* s)
    elseif do_normalize
        return (x .- channel_stats(x, image_mean, "image_mean")) ./
               channel_stats(x, image_std, "image_std")
    end
    return do_rescale ? x .* Float32(rescale_factor) : x
end

# The torchvision backend would resize by `shortest_edge` or `max_height` first, which
# aren't ported.
function size_hw(size)
    size isa NamedTuple && issetequal(keys(size), (:height, :width)) &&
        all(v -> v isa Real && isinteger(v) && v > 0, values(size)) ||
        throw(ArgumentError("`size` must have `height` and `width` only, as positive integers, got $(repr(size))"))
    return Int(size.height), Int(size.width)
end

# transformers' `get_resize_output_image_size` of a `shortest_edge`: the `(height, width)`
# whose shorter side is `shortest_edge`, the longer scaled alike, rounded down.
function get_resize_output_image_size(height, width, shortest_edge)
    short, long = minmax(height, width)
    new_long = trunc(Int, shortest_edge * long / short)
    return width <= height ? (new_long, shortest_edge) : (shortest_edge, new_long)
end

# The torchvision backend's `center_crop` of the image `x` (width, height, channels) to
# `(height, width)`: padded with zeros where it is smaller, half the padding (rounded down)
# before, and cropped from the offsets rounded down.
function center_crop(x, (crop_height, crop_width))
    width, height = size(x, 1), size(x, 2)
    if crop_width > width || crop_height > height
        left, top = max(crop_width - width, 0) ÷ 2, max(crop_height - height, 0) ÷ 2
        padded = zeros(eltype(x), max(crop_width, width), max(crop_height, height),
            size(x, 3))
        padded[left .+ (1:width), top .+ (1:height), :] = x
        x = padded
        width, height = size(x, 1), size(x, 2)
    end
    top, left = (height - crop_height) ÷ 2, (width - crop_width) ÷ 2
    return x[left .+ (1:crop_width), top .+ (1:crop_height), :]
end

# An image as the torchvision backend prepares it for the image processor `ip`, resized to
# the `(height, width)` that `target_size(height, width)` gives, and center-cropped to
# `crop_size`, `(height, width)`, unless it is `nothing`.
function prepare_image(ip, image, target_size; crop_size = nothing)
    x = ip.do_convert_rgb ? convert_to_rgb(image) : image
    eightbit = is_8bit(x)
    if ip.do_resize
        h, w = target_size(size(x, 2), size(x, 1))
        x = resize_image(x, (w, h), ip.resample)
    else
        x = Float32.(x)
    end
    crop_size === nothing || (x = center_crop(x, crop_size))
    return rescale_and_normalize(x, eightbit, ip.do_rescale, ip.rescale_factor,
        ip.do_normalize, ip.image_mean, ip.image_std)
end

function stack_images(images)
    allequal(size.(images)) ||
        error("images of different sizes $(unique(size.(images))) can't be batched; resize them (`do_resize = true`)")
    return stack(images)
end

# The sizes of images, such as `target_sizes`, as a vector of `(; height, width)`: from a
# collection of `(; height, width)` or of `(height, width)` pairs, or a 2 × batch matrix.
size_list(::Nothing) = nothing
size_list(sizes::AbstractMatrix{<:Integer}) = [image_size(s) for s in eachcol(sizes)]
size_list(sizes) = [image_size(s) for s in sizes]

function image_size(s::NamedTuple)
    haskey(s, :height) && haskey(s, :width) ||
        throw(ArgumentError("an image size is `(; height, width)` or `(height, width)`, got $(repr(s))"))
    return (; height = Int(s.height), width = Int(s.width))
end
function image_size(s)
    s isa Union{Tuple, AbstractVector} && length(s) == 2 ||
        throw(ArgumentError("an image size is `(; height, width)` or `(height, width)`, got $(repr(s))"))
    return (; height = Int(first(s)), width = Int(last(s)))
end
