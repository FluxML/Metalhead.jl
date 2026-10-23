using Documenter, Metalhead

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

function family_file(spec)
    if spec.format == Metalhead.Timm.TIMM
        aliases = Dict("vision_transformer" => "vit", "xception_aligned" => "xception")
        file = get(aliases, spec.family, spec.family)
        return "src/Timm/$file.jl"
    end
    return relpath(string(first(methods(spec.build)).file), pkgdir(Metalhead))
end

function write_model_list(path)
    titles = Dict(
        "convmixer" => "ConvMixer",
        "convnext" => "ConvNeXt",
        "densenet" => "DenseNet",
        "efficientnet" => "EfficientNet, MobileNetV1/V2, MNASNet and MixNet",
        "flux2" => "FLUX.2",
        "inception_resnet_v2" => "Inception-ResNet-v2",
        "inception_v3" => "Inception-v3",
        "inception_v4" => "Inception-v4",
        "llama" => "Llama",
        "mlp_mixer" => "MLP-Mixer, ResMLP and gMLP",
        "mobilenetv3" => "MobileNetV3/V4, FBNetV3 and LCNet",
        "qwen2" => "Qwen2",
        "qwen3" => "Qwen3",
        "res2net" => "Res2Net",
        "resnet" => "ResNet",
        "vgg" => "VGG",
        "vision_transformer" => "Vision Transformer (ViT)",
        "xception_aligned" => "Aligned Xception")
    title(family) = get(titles, family, family)
    ArchSpec = Metalhead.ArchSpec
    groups = Dict{String, Dict{String, Vector{ArchSpec}}}()
    for (family, names) in list_models(; byfamily = true)
        for name in names
            spec = Metalhead.REGISTRY[name]
            by_family = get!(groups, string(spec.format), Dict{String, Vector{ArchSpec}}())
            push!(get!(by_family, family, ArchSpec[]), spec)
        end
    end
    narchs = length(Metalhead.REGISTRY)
    nfamilies = sum(length, values(groups))
    nrefs = sum(length(s.references) for s in values(Metalhead.REGISTRY))
    open(path, "w") do io
        println(io, """
            # [Models](@id models)

            Metalhead has $narchs ported architectures in $nfamilies families. Build one
            with `create_model("<architecture>")`, or load a Hub checkpoint of it with
            `create_model("<organisation>/<repository>"; pretrained = true)`.

            Models are grouped by library and family, following the original library's model
            modules. Wide ResNet, ResNeXt and SE-ResNet belong to the ResNet family;
            Res2Net and aligned Xception have their own families.

            Browse the same families from Julia with [`list_models`](@ref):

            ```julia
            families = list_models(; byfamily = true)
            families["resnet"]
            list_models(; byfamily = true, pretrained = true)["resnet"]
            ```

            The $nrefs checkpoints listed here were compared with PyTorch's outputs (see
            [How models are validated](@ref validation)). Bare architecture names load the
            first one with `pretrained = true`. Other checkpoints of the same architectures
            load with the same strict checks. A dash means no checkpoint has been validated
            for that architecture.

            ```@contents
            Pages = ["models.md"]
            Depth = 3
            ```
            """)
        for format in sort!(collect(keys(groups)))
            println(io, "## $format\n")
            by_family = groups[format]
            for family in sort!(collect(keys(by_family)); by = title)
                specs = sort!(by_family[family]; by = s -> s.name)
                println(io, "### $(title(family))\n")
                count = length(specs)
                noun = count == 1 ? "architecture" : "architectures"
                file = family_file(first(specs))
                println(io, "`$family`: $count $noun. ",
                    "[Source](https://github.com/FluxML/Metalhead.jl/blob/master/$file).\n")
                println(io, "| Architecture | Validated checkpoints |\n|:--|:--|")
                for s in specs
                    refs = join(("[`$repo`](https://huggingface.co/$repo/tree/$sha)"
                                 for (repo, sha) in s.references), ", ")
                    isempty(refs) && (refs = "—")
                    println(io, "| `$(s.name)` | $refs |")
                end
                println(io)
            end
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
        "Vision models" => "tutorials/vision.md",
        "Language models" => "tutorials/language_models.md",
        "Text-to-image generation" => "tutorials/text_to_image.md",
        "Models" => "models.md",
        "How models are validated" => "validation.md",
        "Porting a model" => "contributing.md",
        "Migrating from 0.9" => "migration.md",
        "API reference" => "api/reference.md"
    ],
    warnonly = [:example_block, :missing_docs, :cross_references],
    format = Documenter.HTML(canonical = "https://fluxml.ai/Metalhead.jl/stable/",
        assets = ["assets/flux.css"],
        prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(; repo = "github.com/FluxML/Metalhead.jl.git", target = "build")
