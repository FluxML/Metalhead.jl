"""
    DropBlock(drop_prob = 0.1, block_size = 7)

Implements DropBlock, a regularization method for convolutional networks.
([reference](https://arxiv.org/pdf/1810.12890.pdf))
"""
struct DropBlock{F}
    drop_prob::F
    block_size::Integer
end
@functor DropBlock

(m::DropBlock)(x) = dropblock(x, m.drop_prob, m.block_size)

DropBlock(drop_prob = 0.1, block_size = 7) = DropBlock(drop_prob, block_size)

function _dropblock_checks(x, drop_prob, T)
    if !(T <: AbstractArray)
        throw(ArgumentError("x must be an `AbstractArray`"))
    end
    if ndims(x) != 4
        throw(ArgumentError("x must have 4 dimensions (H, W, C, N) for `DropBlock`"))
    end
    @assert drop_prob < 0 || drop_prob > 1 "drop_prob must be between 0 and 1, got $drop_prob"
end
ChainRulesCore.@non_differentiable _dropblock_checks(x, drop_prob, T)

function dropblock(x::T, drop_prob, block_size::Integer) where {T}
    _dropblock_checks(x, drop_prob, T)
    if drop_prob == 0
        return x
    end
    return _dropblock(x, drop_prob, block_size)
end

function _dropblock(x::AbstractArray{T, 4}, drop_prob, block_size) where {T}
    gamma = drop_prob / (block_size ^ 2)
    mask = rand_like(x, Float32, (size(x, 1), size(x, 2), size(x, 3)))
    mask .<= gamma
    block_mask = maxpool(reshape(mask, (size(mask)[1:3]..., 1)), (block_size, block_size);
                         pad = block_size ÷ 2, stride = (1, 1))
    if block_size % 2 == 0
        block_mask = block_mask[1:(end - 1), 1:(end - 1), :, :]
    end
    block_mask = 1 .- dropdims(block_mask; dims = 4)
    out = (x .* reshape(block_mask, (size(block_mask)[1:3]..., 1))) * length(block_mask) /
          sum(block_mask)
    return out
end

"""
    DropPath(p)

Implements Stochastic Depth - equivalent to `Dropout(p; dims = 4)` when `p` ≥ 0.
([reference](https://arxiv.org/abs/1603.09382))

# Arguments

  - `p`: rate of Stochastic Depth.
"""
DropPath(p) = p ≥ 0 ? Dropout(p; dims = 4) : identity
