using Metalhead, Test
using Flux
using Metalhead: AbstractConfigFormat, ArchSpec, REGISTRY, register!, lookup, identify,
                 model_source, model_args, skip_prefix
using Metalhead.Timm: TIMM

const FIXTURE = joinpath(pkgdir(Metalhead), "test", "fixtures", "resnet_basic")

@testset "names" begin
    @test_throws "org/name" create_model("resnet50.a1_in1k")
    @test_throws "org/name" create_model("not_a_model"; pretrained = true)
    unvalidated = first(s.name for s in values(REGISTRY) if isempty(s.references))
    @test_throws "no validated" create_model(unvalidated; pretrained = true)
    @test_throws MethodError list_models(; library = :timm)
    @test issorted(list_models())
    @test list_models() isa Vector{String}
    @test length(list_models()) == length(REGISTRY)
end

@testset "families" begin
    families = list_models(; byfamily = true)
    @test families isa Dict{String, Vector{String}}
    @test sort!(reduce(vcat, values(families))) == list_models()
    @test all(issorted, values(families))
    @test !haskey(families, "other")
    @test "wide_resnet50_2" in families["resnet"]
    @test "wide_resnet101_2" in families["resnet"]
    @test "res2net50_26w_4s" in families["res2net"]
    @test "res2next50" in families["res2net"]
    @test "res2net50_26w_4s" ∉ families["resnet"]
    @test families["xception_aligned"] ==
          ["xception41", "xception41p", "xception65", "xception65p", "xception71"]
    @test "vit_tiny_patch16_224" in families["vision_transformer"]
    @test families["inception"] == ["inception_resnet_v2", "inception_v3", "inception_v4"]

    checkpoints = list_models(; byfamily = true, pretrained = true)
    @test checkpoints isa Dict{String, Vector{String}}
    @test sort!(reduce(vcat, values(checkpoints))) == sort(list_models(; pretrained = true))
    @test all(names -> !isempty(names), values(checkpoints))
    @test "timm/wide_resnet50_2.tv2_in1k" in checkpoints["resnet"]
    @test "timm/xception41p.ra3_in1k" in checkpoints["xception_aligned"]

    empty!(families["resnet"])
    @test "resnet18" in list_models(; byfamily = true)["resnet"]
end

@testset "model sources" begin
    repo, sha = first(lookup("resnet18").references)
    s = model_source("resnet18", nothing)
    @test (s.repo, s.revision) == (repo, sha)
    @test model_source("resnet18", "main").revision == "main"
    @test model_source(repo, nothing).revision == sha
    @test model_source(repo, "v1").revision == "v1"
    @test model_source("timm/not_validated.x", nothing).revision == "main"
    s = model_source(FIXTURE, nothing)
    @test s.repo === nothing && s.dir == FIXTURE

    # a component of a diffusers pipeline, in a subfolder of its repository
    pipeline, sha = first(lookup("Flux2KleinPipeline").references)
    s = model_source(pipeline * "/vae", nothing)
    @test (s.repo, s.subfolder, s.revision) == (pipeline, "vae", sha)
    @test Metalhead.join_path(s.subfolder, "config.json") == "vae/config.json"
    s = model_source("AutoencoderKLFlux2", nothing)
    @test (s.repo, s.subfolder) == (pipeline, "vae")
    @test model_source(pipeline, nothing, "transformer").subfolder == "transformer"
    @test model_source("org/name/sub", "v1", "dir").subfolder == "sub/dir"
end

@testset "diffusers formats" begin
    using Metalhead.Diffusers: DIFFUSERS, DIFFUSERS_PIPELINE
    dir = joinpath(pkgdir(Metalhead), "test", "fixtures", "flux2_klein_tiny")
    @test string(DIFFUSERS) == "diffusers" && string(DIFFUSERS_PIPELINE) == "diffusers pipeline"
    s = model_source(dir, nothing)
    @test identify(Metalhead.source_config(s)) == (DIFFUSERS_PIPELINE, "Flux2KleinPipeline")
    vae = Metalhead.subsource(s, "vae")
    @test identify(Metalhead.source_config(vae)) == (DIFFUSERS, "AutoencoderKLFlux2")
    @test Metalhead.source_weights(vae, Metalhead.weights_name(DIFFUSERS)) ==
          [joinpath(dir, "vae", "diffusion_pytorch_model.safetensors")]
    @test_throws "has no modules.json, config.json or model_index.json" Metalhead.source_config(
        Metalhead.subsource(s, "tokenizer"))
end

@testset "timm format" begin
    @test string(TIMM) == "timm"
    config = Metalhead.JSON.parsefile(joinpath(FIXTURE, "config.json"))
    @test identify(config) == (TIMM, "resnet18")
    @test_throws "ported library" identify(Dict("model_type" => "dinov2"))

    spec = lookup("resnet18")
    args = model_args(TIMM, spec, config, (; drop_rate = 0.5))
    @test args.layers == (1, 1, 1, 1) && args.num_classes == 10 && args.drop_rate == 0.5
    @test skip_prefix(TIMM, config, args) === nothing
    @test skip_prefix(TIMM, config, merge(args, (; num_classes = 3))) == "fc"
    @test_throws "in_chans" skip_prefix(TIMM, config, merge(args, (; in_chans = 1)))
    nameless = deepcopy(config)
    delete!(nameless["pretrained_cfg"], "classifier")
    @test_throws "classifier" skip_prefix(TIMM, nameless, merge(args, (; num_classes = 3)))

    cfg = Metalhead.Timm.pretrained_cfg(FIXTURE)
    @test cfg.input_size == (3, 32, 32) && cfg.classifier == "fc"
end

# A second library only has to say how its config.json names an architecture.
struct ToyFormat <: AbstractConfigFormat end
Metalhead.architecture_of(::ToyFormat, config) = get(config, "kind", nothing)
toy(; width = 2) = Dense(width => width)

@testset "another config format" begin
    register!(ArchSpec(ToyFormat(), "toy", toy; family = "toy_family",
        defaults = (; width = 3)))
    try
        @test "toy" in list_models()
        @test list_models(; byfamily = true)["toy_family"] == ["toy"]
        @test !haskey(list_models(; byfamily = true, pretrained = true), "toy_family")
        @test ArchSpec(ToyFormat(), "fallback", toy).family == "other"
        @test_throws "already registered" register!(ArchSpec(TIMM, "toy", toy))
        @test size(create_model("toy").weight) == (3, 3)
        mktempdir() do dir
            write(joinpath(dir, "config.json"), """{"kind": "toy"}""")
            config = Metalhead.JSON.parsefile(joinpath(dir, "config.json"))
            @test identify(config)[2] == "toy"
            # a directory names a checkpoint, whose weights load by default
            @test_throws "has no safetensors weights" create_model(dir)
            @test size(create_model(dir; pretrained = false).weight) == (3, 3)
            @test size(create_model(dir; pretrained = false, width = 4).weight) == (4, 4)
            # Third-party builders keep the original build-and-convert contract.
            Metalhead.SafeTensors.serialize(joinpath(dir, "model.safetensors"),
                Dict("weight" => ones(Float32, 3, 3), "bias" => zeros(Float32, 3)))
            for T in (Float16, Flux.BFloat16, Float32, Float64)
                model = create_model(dir; eltype = T)
                @test eltype(model.weight) == T
                @test model.weight == ones(T, 3, 3)
            end
            write(joinpath(dir, "config.json"), """{"kind": "missing"}""")
            @test_throws "not been ported" create_model(dir; pretrained = false)
        end
    finally
        delete!(REGISTRY, "toy")
    end
    @test "toy" ∉ list_models()
end
