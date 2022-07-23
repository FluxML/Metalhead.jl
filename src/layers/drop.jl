# Generates the mask to be used for `DropBlock`
@inline function _dropblock_mask(rng, x, gamma, clipped_block_size)
    block_mask = rand_like(rng, x)
    block_mask .= block_mask .< gamma
    return 1 .- maxpool(block_mask, (clipped_block_size, clipped_block_size);
                   stride = 1, pad = clipped_block_size ÷ 2)
end
ChainRulesCore.@non_differentiable _dropblock_mask(rng, x, gamma, clipped_block_size)

"""
    dropblock([rng = rng_from_array(x)], x::AbstractArray{T, 4}, drop_block_prob, block_size,
              gamma_scale, active::Bool = true)

The dropblock function. If `active` is `true`, for each input, it zeroes out continguous
regions of size `block_size` in the input. Otherwise, it simply returns the input `x`.

# Arguments

  - `rng`: can be used to pass in a custom RNG instead of the default. Custom RNGs are only
    supported on the CPU.
  - `x`: input array
  - `drop_block_prob`: probability of dropping a block
  - `block_size`: size of the block to drop
  - `gamma_scale`: multiplicative factor for `gamma` used. For the calculations,
    refer to [the paper](https://arxiv.org/abs/1810.12890).

If you are an end-user, you do not want this function. Use [`DropBlock`](#) instead.
"""
# TODO add experimental `DropBlock` options from timm such as gaussian noise and
# more precise `DropBlock` to deal with edges.
function dropblock(rng::AbstractRNG, x::AbstractArray{T, 4}, drop_block_prob, block_size,
                   gamma_scale) where {T}
    H, W, _, _ = size(x)
    total_size = H * W
    clipped_block_size = min(block_size, min(H, W))
    gamma = gamma_scale * drop_block_prob * total_size / clipped_block_size^2 /
            ((W - block_size + 1) * (H - block_size + 1))
    block_mask = dropblock_mask(rng, x, gamma, clipped_block_size)
    normalize_scale = length(block_mask) / sum(block_mask) .+ T(1e-6)
    return x .* block_mask .* normalize_scale
end

## bs is `clipped_block_size`
# Dispatch for GPU
dropblock_mask(rng::CUDA.RNG, x::CuArray, gamma, bs) = _dropblock_mask(rng, x, gamma, bs)
function dropblock_mask(rng, x::CuArray, gamma, bs)
    throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropblock only supports CUDA.RNG for CuArrays."))
end
# Dispatch for CPU
dropblock_mask(rng, x, gamma, bs) = _dropblock_mask(rng, x, gamma, bs)

mutable struct DropBlock{F, R <: AbstractRNG}
    drop_block_prob::F
    block_size::Integer
    gamma_scale::F
    active::Union{Bool, Nothing}
    rng::R
end
@functor DropBlock
trainable(a::DropBlock) = (;)

function _dropblock_checks(x::AbstractArray{<:Any, 4}, drop_block_prob, gamma_scale)
    @assert 0 ≤ drop_block_prob ≤ 1
    "drop_block_prob must be between 0 and 1, got $drop_block_prob"
    @assert 0 ≤ gamma_scale ≤ 1
    "gamma_scale must be between 0 and 1, got $gamma_scale"
end
function _dropblock_checks(x, drop_block_prob, gamma_scale)
    throw(ArgumentError("x must be an array with 4 dimensions (H, W, C, N) for DropBlock."))
end
ChainRulesCore.@non_differentiable _dropblock_checks(x, drop_block_prob, gamma_scale)

function (m::DropBlock)(x)
    _dropblock_checks(x, m.drop_block_prob, m.gamma_scale)
    if Flux._isactive(m)
        return dropblock(m.rng, x, m.drop_block_prob, m.block_size, m.gamma_scale)
    else
        return x
    end
end

function Flux.testmode!(m::DropBlock, mode = true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

"""
    DropBlock(drop_block_prob = 0.1, block_size = 7, gamma_scale = 1.0,
              rng = rng_from_array())

The `DropBlock` layer. While training, it zeroes out continguous regions of
size `block_size` in the input. During inference, it simply returns the input `x`.
((reference)[https://arxiv.org/abs/1810.12890])

# Arguments

  - `drop_block_prob`: probability of dropping a block
  - `block_size`: size of the block to drop
  - `gamma_scale`: multiplicative factor for `gamma` used. For the calculation of gamma,
    refer to [the paper](https://arxiv.org/abs/1810.12890).
  - `rng`: can be used to pass in a custom RNG instead of the default. Custom RNGs are only
    supported on the CPU.
"""
function DropBlock(drop_block_prob = 0.1, block_size = 7, gamma_scale = 1.0,
                   rng = rng_from_array())
    if drop_block_prob == 0.0
        return identity
    end
    return DropBlock(drop_block_prob, block_size, gamma_scale, nothing, rng)
end

function Base.show(io::IO, d::DropBlock)
    print(io, "DropBlock(", d.drop_block_prob)
    print(io, ", block_size = $(repr(d.block_size))")
    print(io, ", gamma_scale = $(repr(d.gamma_scale))")
    return print(io, ")")
end

"""
    DropPath(p; [rng = rng_from_array(x)])

Implements Stochastic Depth - equivalent to `Dropout(p; dims = 4)` when `0 < p ≤ 1` and
`identity` otherwise.
([reference](https://arxiv.org/abs/1603.09382))

This layer can be used to drop certain blocks in a residual structure and allow them to
propagate completely through the skip connection. It can be used in two ways: either with
all blocks having the same survival probability or with a linear scaling rule across the
blocks. This is performed only at training time. At test time, the `DropPath` layer is
equivalent to `identity`.

!!! warning
    
    In the case of the linear scaling rule, the calculations of survival probabilities for each
    block may lead to a survival probability > 1 for a given block. This will lead to
    `DropPath` returning `identity`, which may not be desirable. This usually happens with
    a low number of blocks and a high base survival probability, so it is recommended to
    use a fixed base survival probability across blocks. If this is not possible, then
    a lower base survival probability is recommended.

# Arguments

  - `p`: rate of Stochastic Depth.
  - `rng`: can be used to pass in a custom RNG instead of the default. See `Flux.Dropout`
    for more information on the behaviour of this argument. Custom RNGs are only supported
    on the CPU.
"""
DropPath(p; rng = rng_from_array()) = 0 < p ≤ 1 ? Dropout(p; dims = 4, rng) : identity
