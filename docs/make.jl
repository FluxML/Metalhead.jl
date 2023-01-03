using Documenter, Metalhead, Artifacts, LazyArtifacts, Images, OneHotArrays, DataAugmentation, Flux

DocMeta.setdocmeta!(Metalhead, :DocTestSetup, :(using Metalhead); recursive = true)

makedocs(modules = [Metalhead, Artifacts, LazyArtifacts, Images, OneHotArrays, DataAugmentation, Flux],
         sitename = "Metalhead.jl",
         doctest = false,
         pages = ["Home" => "index.md",
                  "Tutorials" => [
                      "tutorials/quickstart.md",
                   ],
                  "Developer guide" => "contributing.md",
                  "API reference" => [
                      "api/reference.md",
                   ],
                 ],
         format = Documenter.HTML(
              canonical = "https://fluxml.ai/Metalhead.jl/stable/",
            #   analytics = "UA-36890222-9",
              assets = ["assets/flux.css"],
              prettyurls = get(ENV, "CI", nothing) == "true"),
        )

deploydocs(repo = "github.com/FluxML/Metalhead.jl.git",
           target = "build",
           push_preview = true)
