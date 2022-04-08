"""
    weights(model)

Load the pre-trained weights for `model` using the stored artifacts.
"""
function weights(model)
  try
    path = joinpath(@artifact_str(model), "$model.bson")
    artifact = BSON.load(path, @__MODULE__)
    if haskey(artifact, :model)
      return artifact[:model]
    else
      throw(ArgumentError("No pre-trained weights available for $model."))
    end
  catch e
    throw(ArgumentError("No pre-trained weights available for $model."))
  end
end

"""
    loadpretrain!(model, name)

Load the pre-trained weight artifacts matching `<name>.bson` into `model`.
"""
loadpretrain!(model, name) = Flux.loadmodel!(model, weights(name))
