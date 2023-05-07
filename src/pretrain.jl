"""
    loadweights(artifact_name)

Load the pre-trained weights for `model` using the stored artifacts.
"""
function loadweights(artifact_name)
    artifact_dir = try
        @artifact_str(artifact_name)   
    catch e
        throw(ArgumentError("No pre-trained weights available for $artifact_name."))
    end
    file_name = readdir(artifact_dir)[1]
    file_path = joinpath(artifact_dir, file_name)
    
    if endswith(file_name, ".bson")
        artifact = BSON.load(file_path, @__MODULE__)
        if haskey(artifact, :model_state)
            return artifact[:model_state]
        elseif haskey(artifact, :model)
            return artifact[:model]
        else
            throw(ErrorException("Found weight artifact for $artifact_name but the weights are not saved under the key :model_state or :model."))
        end
    elseif endswith(file_path, ".jld2")
        artifact = JLD2.load(file_path)
        if haskey(artifact, "model_state")
            return artifact["model_state"]
        elseif haskey(artifact, "model")
            return artifact["model"]
        else
            throw(ErrorException("Found weight artifact for $artifact_name but the weights are not saved under the key \"model_state\" or \"model\"."))
        end
    else
        throw(ErrorException("Found weight artifact for $artifact_name but only jld2 and bson serialization format are supported."))
    end
end

"""
    loadpretrain!(model, artifact_name)

Load the pre-trained weight artifacts matching `<name>.bson` into `model`.
"""
function loadpretrain!(model, artifact_name)
    m = loadweights(artifact_name)
    Flux.loadmodel!(model, m)
end
