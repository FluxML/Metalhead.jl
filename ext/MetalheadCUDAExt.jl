module MetalheadCUDAExt

if isdefined(Base, :get_extension)
    using Metalhead: Metalhead
else
    using ..Metalhead: Metalhead
end
using CUDA: CUDA, CuArray

## bs is `clipped_block_size`
# Dispatch for GPU
Metalhead.Layers.dropblock_mask(rng::CUDA.RNG, x::CuArray, gamma, bs) = Metalhead.Layers._dropblock_mask(rng, x, gamma, bs)
function Metalhead.Layers.dropblock_mask(rng, x::CuArray, gamma, bs)
    throw(ArgumentError("x isa CuArray, but rng isa $(typeof(rng)). dropblock only supports
                        CUDA.RNG for CuArrays."))
end

end
