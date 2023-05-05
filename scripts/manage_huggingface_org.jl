using HuggingFaceApi, ArtifactUtils

weight_folders = readdir(joinpath(@__DIR__, "weights"), join=true)
@assert all(isdir, weight_folders)
weight_names = basename.(weight_folders)

# fluxml_model_repos = HuggingFaceApi.list_models(author="FluxML")
# model_repo = fluxml_model_repos[1]
# HuggingFaceApi.list_model_files(model_repo[:id])
# url = string(HuggingFaceApi.HuggingFaceURL(model_repo[:id], "vgg11_IMAGENET1K_V1.jld2"))

