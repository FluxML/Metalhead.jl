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
