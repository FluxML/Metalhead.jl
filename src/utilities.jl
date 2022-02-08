# Utility function for getting chunks of an ND-array along a particular dimension
chunk(A, k::Int; dim::Int = 1) = 
    (selectdim(A, dim, i) for i in Iterators.partition(axes(A,dim), cld(size(A,dim), k)));

# Utility function for classifier head of vision transformer-like models
_seconddimmean(x) = mean(x, dims = 2)[:, 1, :]

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
    weights(model)

Load the pre-trained weights for `model` using the stored artifacts.
"""
function weights(model)
  try
    path = joinpath(@artifact_str(model), "$model.bson")
    return BSON.load(path, @__MODULE__)[:weights]
  catch e
    throw(ArgumentError("No pre-trained weights available for $model."))
  end
end

"""
    loadpretrain!(model, name)

Load the pre-trained weight artifacts matching `<name>.bson` into `model`.
"""
loadpretrain!(model, name) = Flux.loadparams!(model, weights(name))

function _maybe_big_show(io, model)
  if isdefined(Flux, :_big_show)
    if isnothing(get(io, :typeinfo, nothing)) # e.g. top level in REPL
      Flux._big_show(io, model)
    else
      show(io, model)
    end
  else
    show(io, model)
  end
end
