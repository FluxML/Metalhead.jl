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
    if length(readdir(artifact_dir)) > 1
        # @warn("Found multiple files in $artifact_dir.")
        files = readdir(artifact_dir)
        files = filter!(x -> endswith(x, ".bson") || endswith(x, ".jld2"), files)
        files = filter!(x -> !startswith(x, "."), files)
        if length(files) > 1
            throw(ErrorException("Found multiple weight artifacts for $artifact_name."))
        end
        file_name = files[1]
    else
        file_name = readdir(artifact_dir)[1]
    end

    file_path = joinpath(artifact_dir, file_name)

    return load_weights_file(file_path)
end

function load_weights_file(file_path::String)
    if endswith(file_path, ".bson")
        artifact = BSON.load(file_path, @__MODULE__)
        if haskey(artifact, :model_state)
            return artifact[:model_state]
        elseif haskey(artifact, :model)
            return artifact[:model]
        else
            throw(ErrorException("Weights in the file `$file_path` are not saved under the key :model_state or :model."))
        end
    elseif endswith(file_path, ".jld2")
        artifact = JLD2.load(file_path)
        if haskey(artifact, "model_state")
            return artifact["model_state"]
        elseif haskey(artifact, "model")
            return artifact["model"]
        else
            throw(ErrorException("Weights in the file `$file_path` are not saved under the key \"model_state\" or \"model\"."))
        end
    else
        throw(ErrorException("Only jld2 and bson serialization format are supported for weights files."))
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
