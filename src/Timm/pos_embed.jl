# Ported from timm.layers.pos_embed (timm 1.0.30, Apache-2.0, Ross Wightman).

# Keys' cubic convolution kernel (ATen's cubic_convolution1 for |x| <= 1, 2 for |x| < 2).
# PyTorch's CPU kernels are compiled with FMA contraction: the fma calls make the weights
# bit-identical to theirs.
cubic_convolution1(x, a) = fma(fma(a + 2, x, -(a + 3)) * x, x, 1.0)
cubic_convolution2(x, a) = fma(fma(fma(a, x, -5 * a), x, 8 * a), x, -4 * a)

function cubic_filter(x, a)
    x = abs(x)
    x < 1 && return cubic_convolution1(x, a)
    x < 2 && return cubic_convolution2(x, a)
    return zero(x)
end

# The out_size × in_size matrix of one dimension of torch's F.interpolate(mode = "bicubic",
# align_corners = false), computed in Float64 like ATen for Float64 inputs. With antialias,
# ATen's _upsample_bicubic2d_aa (Keys kernel with a = -0.5, stretched by the downsampling
# factor, renormalised over the taps inside the input); without, upsample_bicubic2d
# (a = -0.75, border indices clamped).
function bicubic_weights(T::Type, in_size::Int, out_size::Int; antialias::Bool)
    weights = zeros(Float64, out_size, in_size)
    scale = in_size / out_size
    if antialias
        support = scale >= 1 ? 2 * scale : 2.0
        invscale = scale >= 1 ? 1 / scale : 1.0
        max_interp_size = 2 * ceil(Int, support) + 1
        for i in 1:out_size
            center = scale * (i - 0.5)
            xmin = max(trunc(Int, center - support + 0.5), 0)
            xsize = min(trunc(Int, center + support + 0.5), in_size) - xmin
            xsize = clamp(xsize, 0, max_interp_size)
            total = 0.0
            for j in 0:(xsize - 1)
                w = cubic_filter((j + xmin - center + 0.5) * invscale, -0.5)
                weights[i, xmin + j + 1] = w
                total += w
            end
            if total != 0
                weights[i, (xmin + 1):(xmin + xsize)] ./= total
            end
        end
    else
        a = -0.75
        for i in 1:out_size
            src = fma(scale, i - 0.5, -0.5)
            k = min(floor(Int, Float32(src)), in_size - 1)  # ATen uses floorf
            t = clamp(src - k, 0.0, 1.0)
            coeffs = (cubic_convolution2(t + 1, a), cubic_convolution1(t, a),
                cubic_convolution1(1 - t, a), cubic_convolution2((1 - t) + 1, a))
            for (m, c) in enumerate(coeffs)
                weights[i, clamp(k + m - 2, 0, in_size - 1) + 1] += c
            end
        end
    end
    return T.(weights)
end
ChainRulesCore.@non_differentiable bicubic_weights(::Any, ::Any, ::Any)

# The weights are built on the CPU, then copied into an array of the kind of `x` (e.g. a
# CuArray). Doing it inside a non-differentiable function keeps the copy away from Zygote.
function bicubic_weights_like(x, in_size, out_size, antialias)
    weights = bicubic_weights(eltype(x), in_size, out_size; antialias)
    return copyto!(similar(x, size(weights)), weights)
end
ChainRulesCore.@non_differentiable bicubic_weights_like(::Any, ::Any, ::Any, ::Any)

# timm.layers.resample_abs_pos_embed on the (D, num_prefix_tokens + H * W, 1) Julia layout,
# with sizes in torch's (H, W) order. Unlike timm, it interpolates in the eltype of `posemb`
# rather than in Float32 (porting/common.py does the same for float64 fixtures).
function resample_abs_pos_embed(posemb::AbstractArray{<:Any, 3}, new_size,
        old_size = nothing; num_prefix_tokens::Int = 1, interpolation = "bicubic",
        antialias::Bool = true)
    num_pos_tokens = size(posemb, 2)
    if new_size[1] * new_size[2] + num_prefix_tokens == num_pos_tokens
        isnothing(old_size) && new_size[1] == new_size[2] && return posemb
        !isnothing(old_size) && Tuple(old_size) == Tuple(new_size) && return posemb
    end
    interpolation == "bicubic" ||
        throw(ArgumentError("unsupported interpolation $(repr(interpolation))"))
    if isnothing(old_size)
        hw = isqrt(num_pos_tokens - num_prefix_tokens)
        old_size = (hw, hw)
    end
    (h, w), (new_h, new_w) = old_size, new_size
    D = size(posemb, 1)

    # torch's spatial tokens are row-major (w fastest), so the grid is (D, W, H) here.
    x = reshape(posemb[:, (num_prefix_tokens + 1):end, :], D, w, h)
    if new_w != w
        weights_w = bicubic_weights_like(x, w, new_w, antialias)
        x = reshape(weights_w * reshape(permutedims(x, (2, 1, 3)), w, :), new_w, D, h)
        x = permutedims(x, (2, 1, 3))
    end
    if new_h != h
        weights_h = bicubic_weights_like(x, h, new_h, antialias)
        x = reshape(reshape(x, :, h) * transpose(weights_h), D, new_w, new_h)
    end
    x = reshape(x, D, :, 1)
    num_prefix_tokens > 0 || return x
    return cat(posemb[:, 1:num_prefix_tokens, :], x; dims = 2)
end
