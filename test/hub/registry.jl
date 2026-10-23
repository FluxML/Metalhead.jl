using Metalhead, Test
using Flux
using Metalhead: ConfigFormat, ArchSpec, REGISTRY, register!, lookup, identify,
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
    @test length(list_models()) == length(REGISTRY)
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
struct ToyFormat <: ConfigFormat end
Metalhead.architecture_of(::ToyFormat, config) = get(config, "kind", nothing)
toy(; width = 2) = Dense(width => width)

@testset "another config format" begin
    register!(ArchSpec(ToyFormat(), "toy", toy; defaults = (; width = 3)))
    try
        @test "toy" in list_models()
        @test_throws "already registered" register!(ArchSpec(TIMM, "toy", toy))
        @test size(create_model("toy").weight) == (3, 3)
        mktempdir() do dir
            write(joinpath(dir, "config.json"), """{"kind": "toy"}""")
            config = Metalhead.JSON.parsefile(joinpath(dir, "config.json"))
            @test identify(config)[2] == "toy"
            @test size(create_model(dir).weight) == (3, 3)
            @test size(create_model(dir; width = 4).weight) == (4, 4)
            write(joinpath(dir, "config.json"), """{"kind": "missing"}""")
            @test_throws "not been ported" create_model(dir)
        end
    finally
        delete!(REGISTRY, "toy")
    end
    @test "toy" ∉ list_models()
end
