using Documenter, Metalhead, Artifacts, LazyArtifacts, Images, DataAugmentation, Flux

DocMeta.setdocmeta!(Metalhead, :DocTestSetup, :(using Metalhead); recursive = true)

makedocs(; modules = [Metalhead, Artifacts, LazyArtifacts, Images, DataAugmentation, Flux],
         sitename = "Metalhead.jl",
         doctest = false,
         pages = ["Home" => "index.md",
             "Tutorials" => [
                 "tutorials/quickstart.md",
                 "tutorials/pretrained.md",
             ],
             "API reference" => [
                "Convolutional Neural Networks" => [
                    "api/others.md",
                    "api/inception.md",
                    "api/resnet.md",
                    "api/densenet.md",
                    "api/hybrid.md",
                    "api/layers.md",
                ],
                "Mixers" => [
                    "api/mixers.md",
                ],
                "Vision Transformers" => [
                    "api/vit.md",
                ],
                "api/utilities.md"
             ],
             "How To" => [
                 "howto/resnet.md",
             ],
             "Contributing to Metalhead" => "contributing.md",
         ],
         format = Documenter.HTML(; canonical = "https://fluxml.ai/Metalhead.jl/stable/",
                                  #   analytics = "UA-36890222-9",
                                  assets = ["assets/flux.css"],
                                  prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(; repo = "github.com/FluxML/Metalhead.jl.git", target = "build",
           push_preview = true)
