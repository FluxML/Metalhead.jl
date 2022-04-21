# Utility function for classifier head of vision transformer-like models
seconddimmean(x) = dropdims(mean(x, dims = 2); dims = 2)

# utility function for making sure that all layers have a channel size divisible by 8
# used by MobileNet variants
function _round_channels(channels, divisor, min_value = divisor)
  new_channels = max(min_value, floor(Int, channels + divisor / 2) ÷ divisor * divisor)
  # Make sure that round down does not go down by more than 10%
  return (new_channels < 0.9 * channels) ? new_channels + divisor : new_channels
end

"""
    addrelu(x, y)

Convenience function for `(x, y) -> @. relu(x + y)`.
Useful as the `connection` argument for [`resnet`](#).
See also [`reluadd`](#).
"""
addrelu(x, y) = @. relu(x + y)

"""
    reluadd(x, y)

Convenience function for `(x, y) -> @. relu(x) + relu(y)`.
Useful as the `connection` argument for [`resnet`](#).
See also [`addrelu`](#).
"""
reluadd(x, y) = @. relu(x) + relu(y)

"""
    cat_channels(x, y, zs...)

Concatenate `x` and `y` (and any `z`s) along the channel dimension (third dimension).
Equivalent to `cat(x, y, zs...; dims=3)`.
Convenient reduction operator for use with `Parallel`.
"""
cat_channels(xy...) = inferredcat(xy...; dims = 3)

function inferredcat(xs::T...; dims = :)::T where T <: AbstractArray
  cat(xs...; dims = dims)
end

# `rrule` doesn't infer through `_project` neatly
# function Zygote.ChainRules.rrule(::typeof(inferredcat), xs::T...; dims = :)::T where T <: AbstractArray
#     sz = size.(xs)
#     function inferredcat_pullback(Δ)
#         (Zygote.ChainRules.NoTangent(), makesub(Δ, size(Δ)[dims], sz, dims = dims)...,)
#     end
#     inferredcat(xs...; dims = dims), inferredcat_pullback
# end

Zygote.@adjoint function inferredcat(xs::T...; dims = :) where T <: AbstractArray
  sz = size.(xs)
  lz = length.(xs)
  inferredcat(xs..., dims = dims), Δ -> (partition_grad(Δ, size(Δ)[dims], sz, dims = dims)...,)
end

function partition_grad(d::AbstractArray{T,N}, x, sz; dims = :) where {T,N}
  sizeatdim = map(x -> x[dims], sz)
    x_start = 1
    m = map(enumerate(sizeatdim)) do (i, ix)
      x_stop = x_start + ix - 1
      p::Base.UnitRange{Int64} = x_start:x_stop
      x_start = x_start + ix
      p
    end
    function gen_indices(m, sz)
      map(m, sz) do m, sz
        ntuple(x -> x == dims ? m : 1:sz[x], N)
      end
    end
    ix = gen_indices(m, sz)
    map(ix_ -> @view(d[ix_...]), ix)
end

"""
    swapdims(perm)

Convenience function for permuting the dimensions of an array.
`perm` is a vector or tuple specifying a permutation of the input dimensions.
Equivalent to `permutedims(x, perm)`.
"""
swapdims(perm) = Base.Fix2(permutedims, perm)

# Utility function for pretty printing large models
function _maybe_big_show(io, model)
  if isdefined(Flux, :_big_show)
    if isnothing(get(io, :typeinfo, nothing)) # e.g. top level in REPL
      Flux._big_show(io, model)
    else
      show(io, model)
    end
  end
end
