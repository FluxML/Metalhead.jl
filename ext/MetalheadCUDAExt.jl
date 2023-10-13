module MetalheadCUDAExt

if isdefined(Base, :get_extension)
    using Metalhead
    using Metalhead: Metalhead, _dropblock_mask
    using CUDA: CUDA, CuArray
else
    using ..Metalhead
    using ..Metalhead: Metalhead, _dropblock_mask
    using ..CUDA: CUDA, CuArray
end

## bs is `clipped_block_size`
# Dispatch for GPU
Metalhead.dropblock_mask(rng::CUDA.RNG, x::CuArray, gamma, bs) = _dropblock_mask(rng, x, gamma, bs)
function Metalhead.dropblock_mask(rng, x::CuArray, gamma, bs)
    throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropblock only supports
                        CUDA.RNG for CuArrays."))
end

end
