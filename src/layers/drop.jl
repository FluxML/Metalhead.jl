using Flux:Zygote

istraining() = false

Zygote.@adjoint istraining() = true, _ -> nothing

_isactive(m) = isnothing(m.active) ? istraining() : m.active

function drop_path(x, drop_prob::Float32 = 0.0f32, scale_by_keep = true; active = false)
  if drop_prob == 0.0f32 || active == false
    return x
  end
  keep_prob = 1 - drop_prob
  shape = tuple(size(x, 1), ntuple(x -> 1, ndims(x) - 1)...)
  random_tensor = rand(Float32, shape) .< keep_prob
  if keep_prob > 0.0f32 && scale_by_keep
    random_tensor = random_tensor / keep_prob
  end
  return x .* random_tensor
end

struct DropPath{F}
  p::F
  scale_by_keep::Bool
  active::Union{Bool, Nothing}
end

"""
Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
"""
function DropPath(p; scale_by_keep = true)
  @assert 0 ≤ p < 1 "p must be in [0, 1)"
  DropPath(p, scale_by_keep, nothing)
end

(m::DropPath)(x) = drop_path(x, m.p, m.scale_by_keep; active = _isactive(m))

@functor DropPath

testmode!(m::DropPath, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)