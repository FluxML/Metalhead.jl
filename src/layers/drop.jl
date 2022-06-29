function dropblock(rng::AbstractRNG, x::AbstractArray{T, 4}, drop_block_prob, block_size,
                   gamma_scale, active::Bool = true) where {T}
    active || return x
    H, W, _, _ = size(x)
    total_size = H * W
    clipped_block_size = min(block_size, min(H, W))
    gamma = gamma_scale * drop_block_prob * total_size / clipped_block_size^2 /
            ((W - block_size + 1) * (H - block_size + 1))
    block_mask = rand_like(rng, x) .< gamma
    block_mask = maxpool(block_mask, (clipped_block_size, clipped_block_size);
                         stride = 1, pad = clipped_block_size ÷ 2)
    block_mask = 1 .- block_mask
    normalize_scale = convert(T, (length(block_mask) / sum(block_mask) .+ 1e-6))
    return x .* block_mask .* normalize_scale
end
dropoutblock(rng::CUDA.RNG, x::CuArray, p, args...) = dropblock(rng, x, p, args...)
function dropblock(rng, x::CuArray, p, args...)
    throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropblock only support CUDA.RNG for CuArrays."))
end

struct DropBlock{F, R <: AbstractRNG}
    drop_block_prob::F
    block_size::Integer
    gamma_scale::F
    active::Union{Bool, Nothing}
    rng::R
end

@functor DropBlock
trainable(a::DropBlock) = (;)

function _dropblock_checks(x::T) where {T}
    if !(T <: AbstractArray)
        throw(ArgumentError("x must be an `AbstractArray`"))
    end
    if ndims(x) != 4
        throw(ArgumentError("x must have 4 dimensions (H, W, C, N) for `DropBlock`"))
    end
end
ChainRulesCore.@non_differentiable _dropblock_checks(x)

function (m::DropBlock)(x)
    _dropblock_checks(x)
    Flux._isactive(m) || return x
    return dropblock(m.rng, x, m.drop_block_prob, m.block_size, m.gamma_scale)
end

function Flux.testmode!(m::DropBlock, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function DropBlock(drop_block_prob = 0.1, block_size = 7, gamma_scale = 1.0,
                   rng = Flux.rng_from_array())
    if drop_block_prob == 0.0
        return identity
    end
    @assert 0 ≤ drop_block_prob ≤ 1
    "drop_block_prob must be between 0 and 1, got $drop_block_prob"
    @assert 0 ≤ gamma_scale ≤ 1
    "gamma_scale must be between 0 and 1, got $gamma_scale"
    return DropBlock(drop_block_prob, block_size, gamma_scale, nothing, rng)
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
