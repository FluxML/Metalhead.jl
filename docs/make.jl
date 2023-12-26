using Documenter, Metalhead

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

makedocs(; modules = [Metalhead],
         sitename = "Metalhead.jl",
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
                "Layers" => [
                    "api/layers_intro.md",
                    "api/layers_adv.md"],
                "Model Utilities" => "api/utilities.md",
             ],
         ],
         warnonly = [:example_block, :missing_docs, :cross_references],
         format = Documenter.HTML(canonical = "https://fluxml.ai/Metalhead.jl/stable/",
                                  #   analytics = "UA-36890222-9",
                                  assets = ["assets/flux.css"],
                                  prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(; repo = "github.com/FluxML/Metalhead.jl.git", target = "build",
           push_preview = true)
