# Images of colorants (`load("image.png")` with FileIO: a `Matrix{RGB{N0f8}}` of size
# `(height, width)`) as the processors take images, `(width, height, channels)` arrays.
module MetalheadImageCoreExt

using Metalhead: Metalhead
using ImageCore: Colorant, AbstractRGB, TransparentRGB, AbstractGray, TransparentGray,
                 FixedPoint, N0f8, red, green, blue, alpha, gray

# The channels of a colorant in the order of Pillow's modes RGB, RGBA, L and LA, whatever the
# order of its fields (BGR, ARGB, ...).
channels(::Type{<:AbstractRGB}) = (red, green, blue)
channels(::Type{<:TransparentRGB}) = (red, green, blue, alpha)
channels(::Type{<:AbstractGray}) = (gray,)
channels(::Type{<:TransparentGray}) = (gray, alpha)
function channels(::Type{C}) where {C}
    throw(ArgumentError("images of $C are not supported; convert them to RGB or Gray first, e.g. `RGB.(img)`"))
end

# 8-bit values stay 8-bit (`N0f8` and `Bool`, as Pillow's modes "L" and "1"), so that they take
# the exact path of the processors; others become reals from 0 to 1.
value_type(::Type{N0f8}) = UInt8
value_type(::Type{Bool}) = UInt8
value_type(::Type{T}) where {T <: FixedPoint} = typeof(float(zero(T)))
value_type(::Type{T}) where {T <: AbstractFloat} = T
value_type(::Type{T}) where {T} = throw(ArgumentError("colorants of $T are not supported"))

value(::Type{UInt8}, x::N0f8) = reinterpret(UInt8, x)
value(::Type{UInt8}, x::Bool) = x ? 0xff : 0x00
value(::Type{T}, x) where {T} = T(x)

# `permutedims(channelview(img), (3, 2, 1))`, in one pass, with the channels in a fixed order
function Metalhead.image_array(img::AbstractMatrix{C}) where {C <: Colorant}
    fs = channels(C)
    T = value_type(eltype(C))
    out = Array{T}(undef, size(img, 2), size(img, 1), length(fs))
    fill_channels!(out, img, fs)
    return out
end

function fill_channels!(out::Array{T}, img, fs) where {T}
    @inbounds for (i, row) in enumerate(axes(img, 1)), (j, col) in enumerate(axes(img, 2))
        c = img[row, col]
        vals = map(f -> value(T, f(c)), fs)
        for k in eachindex(vals)
            out[j, i, k] = vals[k]
        end
    end
    return out
end

end # module
