using Documenter, Metalhead

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

# The model list is generated from the registry, grouped by builder.
function write_model_list(path)
    groups = Dict{Symbol, Vector{Metalhead.ArchSpec}}()
    for spec in values(Metalhead.REGISTRY)
        push!(get!(groups, nameof(spec.build), Metalhead.ArchSpec[]), spec)
    end
    narchs = length(Metalhead.REGISTRY)
    nrefs = sum(length(s.references) for s in values(Metalhead.REGISTRY))
    open(path, "w") do io
        println(io, """
            # [Models](@id models)

            Metalhead has $narchs ported timm architectures. Build one with
            `create_model("<architecture>")`, or load any Hub checkpoint of it with
            `create_model("timm/<architecture>.<tag>"; pretrained = true)`.

            The $nrefs checkpoints listed here were compared with PyTorch's outputs (see
            [How models are validated](@ref validation)). Bare architecture names load the first
            one with `pretrained = true`. Other checkpoints of the same architectures, such as
            those in the [timm organisation](https://huggingface.co/timm), load with the same
            strict checks.
            """)
        for builder in sort!(collect(keys(groups)))
            specs = sort!(groups[builder]; by = s -> s.name)
            file = basename(string(first(methods(first(specs).build)).file))
            println(io, "## `$builder`\n")
            println(io, "Built by `Metalhead.Timm.$builder` ",
                "([source](https://github.com/FluxML/Metalhead.jl/blob/master/src/Timm/$file)).\n")
            println(io, "| Architecture | Validated checkpoints |\n|:--|:--|")
            for s in specs
                refs = join(("[`$repo`](https://huggingface.co/$repo/tree/$sha)"
                             for (repo, sha) in s.references), ", ")
                println(io, "| `$(s.name)` | $refs |")
            end
            println(io)
        end
    end
end
write_model_list(joinpath(@__DIR__, "src", "models.md"))

# Every timm name in the migration guide must be a ported architecture.
let guide = read(joinpath(@__DIR__, "src", "migration.md"), String)
    names = [first(split(replace(m[1], "timm/" => ""), '.'))
             for m in eachmatch(r"\"((?:timm/)?[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)?)\"", guide)]
    unported = setdiff(names, list_models())
    isempty(unported) ||
        error("the migration guide names architectures that are not ported: $unported")
end

makedocs(; modules = [Metalhead],
    sitename = "Metalhead.jl",
    pages = ["Home" => "index.md",
        "Getting started" => "tutorials/quickstart.md",
        "Models" => "models.md",
        "How models are validated" => "validation.md",
        "Porting a model" => "contributing.md",
        "Migrating from 0.9" => "migration.md",
        "API reference" => "api/reference.md",
        "Deprecated API (0.9)" => [
            "Using ResNets" => "howto/resnet.md",
            "Convolutional Neural Networks" => [
                "api/resnet.md",
                "api/densenet.md",
                "api/efficientnet.md",
                "api/mobilenet.md",
                "api/inception.md",
                "api/hybrid.md",
                "api/others.md"
            ],
            "Mixers" => "api/mixers.md",
            "Vision Transformers" => "api/vit.md",
            "Layers" => ["api/layers_intro.md", "api/layers_adv.md"],
            "Model Utilities" => "api/utilities.md"
        ]
    ],
    warnonly = [:example_block, :missing_docs, :cross_references],
    format = Documenter.HTML(canonical = "https://fluxml.ai/Metalhead.jl/stable/",
        assets = ["assets/flux.css"],
        prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(; repo = "github.com/FluxML/Metalhead.jl.git", target = "build")
