# Ported from timm.layers.pos_embed (timm 1.0.30, Apache-2.0, Ross Wightman).

# timm.layers.resample_abs_pos_embed on the (D, num_prefix_tokens + H * W, 1) Julia layout,
# with sizes in torch's (H, W) order. Like timm, it interpolates half precision in Float32,
# but Float64 in Float64 (porting/common.py does the same for float64 fixtures).
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
    T, Tc = eltype(posemb), promote_type(eltype(posemb), Float32)
    x = permutedims(T === Tc ? x : Tc.(x), (2, 3, 1))
    x = upsample_bicubic(reshape(x, w, h, D, 1); size = (new_w, new_h), antialias)
    x = permutedims(reshape(x, new_w, new_h, D), (3, 1, 2))
    x = reshape(T === Tc ? x : T.(x), D, :, 1)
    num_prefix_tokens > 0 || return x
    return cat(posemb[:, 1:num_prefix_tokens, :], x; dims = 2)
end
