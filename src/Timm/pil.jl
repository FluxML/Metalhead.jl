# Pillow's `Image.resize` of 8-bit images (Pillow 13.0.0: src/libImaging/Resample.c, and
# ImagingScaleAffine in Geometry.c for NEAREST), which timm's eval transform calls through
# torchvision. Images are `(width, height, channels)` arrays of `UInt8`, each channel
# resized on its own, as Pillow does the bands of an "RGB" or "L" image.
#
# Pillow resizes in two separable passes, each rounding to 8 bits, with the weights of each
# output pixel normalised, then rounded to fixed point with `PIL_PRECISION_BITS` bits. The
# order of the passes changes the result, so it follows Pillow's. torchvision's resize of
# `uint8` tensors (torch's UpSampleKernelAVXAntialias.h, from Pillow-SIMD) differs only in
# the order and the precision; `pil_resize(...; torchvision = true)` follows it.

const PIL_PRECISION_BITS = 32 - 8 - 2

pil_box(x) = -0.5 < x <= 0.5 ? 1.0 : 0.0
pil_bilinear(x) = (x = abs(x); x < 1 ? 1 - x : 0.0)

function pil_hamming(x)
    x = abs(x)
    x == 0 && return 1.0
    x >= 1 && return 0.0
    x *= π
    return sin(x) / x * (0.54f0 + 0.46f0 * cos(x))  # Pillow's Float32 constants
end

function pil_bicubic(x)
    a = -0.5
    x = abs(x)
    x < 1 && return ((a + 2) * x - (a + 3)) * x * x + 1
    x < 2 && return (((x - 5) * x + 8) * x - 4) * a
    return 0.0
end

pil_sinc(x) = x == 0 ? 1.0 : (x *= π; sin(x) / x)
pil_lanczos(x) = -3 <= x < 3 ? pil_sinc(x) * pil_sinc(x / 3) : 0.0

# The filter and its support of each of Pillow's resampling filters.
const PIL_FILTERS = Dict(:box => (pil_box, 0.5), :bilinear => (pil_bilinear, 1.0),
    :hamming => (pil_hamming, 1.0), :bicubic => (pil_bicubic, 2.0),
    :lanczos => (pil_lanczos, 3.0))
const PIL_RESAMPLING = (:nearest, :box, :bilinear, :hamming, :bicubic, :lanczos)

# Pillow's precompute_coeffs: for each output pixel, the offset of its first input pixel,
# the number of input pixels and their normalised weights (a column of `k`).
function pil_coeffs(insize, outsize, filter, support)
    scale = insize / outsize
    filterscale = max(scale, 1.0)
    support *= filterscale
    ksize = ceil(Int, support) * 2 + 1
    inv_filterscale = 1.0 / filterscale
    offset, count, k = zeros(Int, outsize), zeros(Int, outsize), zeros(ksize, outsize)
    for i in 1:outsize
        center = (i - 0.5) * scale
        lo = max(trunc(Int, center - support + 0.5), 0)
        n = clamp(min(trunc(Int, center + support + 0.5), insize) - lo, 0, ksize)
        total = 0.0
        for j in 1:n
            k[j, i] = filter((j - 1 + lo - center + 0.5) * inv_filterscale)
            total += k[j, i]
        end
        iszero(total) || (k[1:n, i] ./= total)
        offset[i], count[i] = lo, n
    end
    return offset, count, k
end

# The weights in fixed point, rounded half away from zero, as `normalize_coeffs_8bpc`.
function fixed_point(k, bits)
    scale = 2.0^bits
    return map(w -> unsafe_trunc(Int32, w < 0 ? -0.5 + w * scale : 0.5 + w * scale), k)
end

clip8(s, bits) = UInt8(clamp(s >> bits, 0, 255))

function resample_horizontal(img, (offset, count, k), bits)
    out = similar(img, length(offset), size(img, 2), size(img, 3))
    half = Int32(1) << (bits - 1)
    @inbounds for c in axes(img, 3), y in axes(img, 2), i in eachindex(offset)
        s = half
        for j in 1:count[i]
            s += Int32(img[offset[i] + j, y, c]) * k[j, i]
        end
        out[i, y, c] = clip8(s, bits)
    end
    return out
end

function resample_vertical(img, (offset, count, k), bits)
    out = similar(img, size(img, 1), length(offset), size(img, 3))
    half = Int32(1) << (bits - 1)
    @inbounds for c in axes(img, 3), i in eachindex(offset), x in axes(img, 1)
        s = half
        for j in 1:count[i]
            s += Int32(img[x, offset[i] + j, c]) * k[j, i]
        end
        out[x, i, c] = clip8(s, bits)
    end
    return out
end

# The precision of torchvision's int16 weights (`_compute_index_ranges_int16_weights` in
# torch): the most bits, up to 22, that keep twice the largest weight under 2^15.
function torch_precision(k)
    wmax = maximum(k; init = 0.0)
    bits = 0
    while bits < 22 && unsafe_trunc(Int32, 0.5 + wmax * 2.0^(bits + 1)) < 2^15
        bits += 1
    end
    return bits
end

"""
    pil_resize(img, (width, height), filter = :bicubic; torchvision = false)
        -> Array{UInt8, 3}

`img`, a `(width, height, channels)` array of `UInt8`, resized as Pillow's
`Image.resize((width, height), filter)` resizes an 8-bit image, bit for bit. `filter` is one
of `:nearest`, `:box`, `:bilinear`, `:hamming`, `:bicubic` and `:lanczos`.

With `torchvision = true`, for `:bilinear` and `:bicubic`, it is resized bit for bit as
torchvision resizes a `uint8` tensor on an x86 CPU, with `antialias = True`. torch's variant
of the algorithm, from Pillow-SIMD, always resizes the width first, and rounds the weights
of each pass to as many bits as int16 holds. (For `:bilinear` without AVX2, and on a GPU,
torchvision resizes in floating point instead.)
"""
function pil_resize(img::AbstractArray{UInt8, 3}, (w, h)::NTuple{2, Integer},
        filter::Symbol = :bicubic; torchvision::Bool = false)
    supported = torchvision ? (:bilinear, :bicubic) : PIL_RESAMPLING
    filter in supported ||
        throw(ArgumentError("resampling filter `$filter` is not supported (supported: \
                             $(join(supported, ", ")))"))
    w >= 1 && h >= 1 || throw(ArgumentError("the size must be positive, got $((w, h))"))
    img = Array(img)
    w_in, h_in = size(img)
    (w, h) == (w_in, h_in) && return img
    filter === :nearest && return pil_nearest(img, w, h)
    f, support = PIL_FILTERS[filter]
    function pass(resample, insize, outsize)
        insize == outsize && return identity
        offset, count, k = pil_coeffs(insize, outsize, f, support)
        bits = torchvision ? torch_precision(k) : PIL_PRECISION_BITS
        return x -> resample(x, (offset, count, fixed_point(k, bits)), bits)
    end
    horizontal = pass(resample_horizontal, w_in, w)
    vertical = pass(resample_vertical, h_in, h)
    # Pillow shrinks the height first when it shrinks more than twice as much as the width
    horizontal_first = torchvision || !(h_in - h > 0 && h_in - h > 2 * (w_in - w))
    return horizontal_first ? vertical(horizontal(img)) : horizontal(vertical(img))
end

# Pillow's NEAREST resize, an affine transform (ImagingScaleAffine): the input pixel under
# the centre of each output pixel, the positions accumulated as Pillow accumulates them.
function pil_nearest(img, w, h)
    w_in, h_in, c = size(img)
    function sources(insize, outsize)
        a = insize / outsize
        pos, src = a * 0.5, zeros(Int, outsize)
        for i in 1:outsize
            j = pos < 0 ? -1 : unsafe_trunc(Int, pos)
            src[i] = 0 <= j < insize ? j + 1 : 0
            pos += a
        end
        return src
    end
    xs, ys = sources(w_in, w), sources(h_in, h)
    out = zeros(UInt8, w, h, c)
    for ch in 1:c, (y, yi) in enumerate(ys), (x, xi) in enumerate(xs)
        xi > 0 && yi > 0 && (out[x, y, ch] = img[xi, yi, ch])
    end
    return out
end
