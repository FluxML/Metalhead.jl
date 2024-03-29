using HuggingFaceApi, ArtifactUtils
using PythonCall
const hfhub = pyimport("huggingface_hub")

"""
List the models available in the HuggingFace repos
and add them to the Artifacts.toml file.
"""
function generate_artifacts_toml(model_repos)
    for model_repo in model_repos
        model_files = HuggingFaceApi.list_model_files(model_repo[:id])
        git_refs = hfhub.list_repo_refs(model_repo[:id])
        commit = git_refs.branches[0].target_commit |> string
        artifact_urls = String[]
        for file in model_files
            if endswith(file, ".tar") || endswith(file, ".tar.gz")
                url = HuggingFaceURL(model_repo[:id], file) |> string
                url = replace(url, "main" => commit) # pinpoint to commit instead of branch so that a specific metalhead version
                                                     # points to a specific weights version
                push!(artifact_urls, url)
            end
        end

        for artifact_url in artifact_urls
            artifact_name = split(basename(artifact_url), ".")[1]
            artifact_name = string(artifact_name) # substring to string
            @info "Adding artifact $artifact_name from $artifact_url"
            add_artifact!(joinpath(@__DIR__, "Artifacts.toml"),
                            artifact_name,
                            artifact_url,
                            force = true,
                            lazy = true)
        end
    end
end

"""
List all model repos in the FluxML HuggingFace org
"""
list_fluxml_models() = HuggingFaceApi.list_models(author="FluxML")

"""
Create artifacts tarballs from the weights folders
"""
function create_model_artifacts(; force=false)
    model_folders = readdir(joinpath(@__DIR__, "weights"), join=true)
    model_folders = [folder for folder in model_folders if isdir(folder)]
    artifacts = []
    for model_folder in model_folders
        model_name = basename(model_folder)
        weight_folders = readdir(model_folder, join=true)
        weight_folders = [folder for folder in weight_folders if isdir(folder)]
        for weight_folder in weight_folders
            weight_name = basename(weight_folder)
            artifact_path = joinpath(model_folder, "$(weight_name).tar.gz")
            if !isfile(artifact_path) || force
                run(`tar -czvf $(artifact_path) -C $(weight_folder) .`)
            end
            push!(artifacts, (model_name, weight_name, artifact_path))
        end
    end
    return artifacts
end

function upload_artifacts_to_hf(model_artifacts)
    fluxml_model_repos = list_fluxml_models()
    fluxml_repo_names = [split(repo[:id], "/")[2] for repo in fluxml_model_repos]
    for (model_name, weight_name, artifact_path) in model_artifacts
        idx = findfirst(x -> x[:id] == "FluxML/$model_name", fluxml_model_repos)
        if idx === nothing # TODO create_repo if not exists https://huggingface.co/docs/huggingface_hub/v0.14.1/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo
            @warn "Repo $model_name does not exist, skipping..."
            continue
        end
        repo = fluxml_model_repos[idx]
        hfhub.upload_file(path_or_fileobj = artifact_path, 
                                path_in_repo = basename(artifact_path), 
                                repo_id = repo[:id],
                                repo_type = "model",
                                commit_message = "update weights")
    end
    return nothing
end

### Create artifacts and upload to HuggingFace repos ############
# hfhub.login(ENV["HUGGINGFACE_TOKEN"])
# model_artifacts = create_model_artifacts(force=false)
# model_artifacts = filter(model_artifacts) do x
#     startswith(x[1], "resnet152")
# end
# upload_artifacts_to_hf(model_artifacts)

### Generate Artifacts.toml from HuggingFace repos #############
fluxml_model_repos = list_fluxml_models()
# fluxml_model_repos = filter(fluxml_model_repos) do repo
#     name = split(repo[:id], "/")[2]
#     startswith(name, "resnet152")
# end
generate_artifacts_toml(fluxml_model_repos)
