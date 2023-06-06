using Documenter, Metalhead, Artifacts, LazyArtifacts, Images, DataAugmentation, Flux

DocMeta.setdocmeta!(Metalhead, :DocTestSetup, :(using Metalhead); recursive = true)

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

makedocs(; modules = [Metalhead, Artifacts, LazyArtifacts, Images, DataAugmentation, Flux],
         sitename = "Metalhead.jl",
         doctest = false,
         pages = ["Home" => "index.md",
             "Tutorials" => [
                 "tutorials/quickstart.md",
                 "tutorials/pretrained.md",
             ],
             "Guides" => [
                 "howto/resnet.md",
             ],
             "Contributing to Metalhead" => "contributing.md",
             "API reference" => [
                "Convolutional Neural Networks" => [
                    "api/resnet.md",
                    "api/densenet.md",
                    "api/efficientnet.md",
                    "api/mobilenet.md",
                    "api/inception.md",
                    "api/hybrid.md",
                    "api/others.md",
                    ],
                "Mixers" => [
                    "api/mixers.md",
                    ],
                "Vision Transformers" => [
                    "api/vit.md",
                    ],
                "Layers" => "api/layers.md",
                "Model Utilities" => "api/utilities.md",
             ],
         ],
         format = Documenter.HTML(; canonical = "https://fluxml.ai/Metalhead.jl/stable/",
                                  #   analytics = "UA-36890222-9",
                                  assets = ["assets/flux.css"],
                                  prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(; repo = "github.com/FluxML/Metalhead.jl.git", target = "build",
           push_preview = true)
