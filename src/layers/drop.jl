"""
    DropBlock(drop_block_prob = 0.1, block_size = 7, gamma_scale = 1.0)

Implements DropBlock, a regularization method for convolutional networks.
([reference](https://arxiv.org/pdf/1810.12890.pdf))
"""
struct DropBlock{F}
    drop_block_prob::F
    block_size::Integer
    gamma_scale::F
end
@functor DropBlock

(m::DropBlock)(x) = dropblock(x, m.drop_block_prob, m.block_size, m.gamma_scale)

function DropBlock(drop_block_prob = 0.1, block_size = 7, gamma_scale = 1.0)
    if drop_block_prob == 0.0
        return identity
    end
    @assert drop_block_prob < 0 || drop_block_prob > 1
    "drop_block_prob must be between 0 and 1, got $drop_block_prob"
    @assert gamma_scale < 0 || gamma_scale > 1
    "gamma_scale must be between 0 and 1, got $gamma_scale"
    return DropBlock(drop_block_prob, block_size, gamma_scale)
end

function _dropblock_checks(x::T) where {T}
    if !(T <: AbstractArray)
        throw(ArgumentError("x must be an `AbstractArray`"))
    end
    if ndims(x) != 4
        throw(ArgumentError("x must have 4 dimensions (H, W, C, N) for `DropBlock`"))
    end
end
ChainRulesCore.@non_differentiable _dropblock_checks(x)

function dropblock(x::AbstractArray{T, 4}, drop_block_prob, block_size,
                   gamma_scale) where {T}
    _dropblock_checks(x)
    H, W, _, _ = size(x)
    total_size = H * W
    clipped_block_size = min(block_size, min(H, W))
    gamma = gamma_scale * drop_block_prob * total_size / clipped_block_size^2 /
            ((W - block_size + 1) * (H - block_size + 1))
    block_mask = rand_like(x) .< gamma
    block_mask = maxpool(block_mask, (clipped_block_size, clipped_block_size);
                         stride = 1, pad = clipped_block_size ÷ 2)
    block_mask = 1 .- block_mask
    normalize_scale = convert(T, (length(block_mask) / sum(block_mask) .+ 1e-6))
    return x .* block_mask .* normalize_scale
end

"""
    DropPath(p)

Implements Stochastic Depth - equivalent to `Dropout(p; dims = 4)` when `p` ≥ 0 and
`identity` otherwise.
([reference](https://arxiv.org/abs/1603.09382))

# Arguments

  - `p`: rate of Stochastic Depth.
"""
DropPath(p) = p > 0 ? Dropout(p; dims = 4) : identity
