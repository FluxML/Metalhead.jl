using Metalhead, Test
using Flux, SafeTensors
using Metalhead: HF, JSON, ModelSource, source_config, source_weights, hf_download_first

const FIXTURES = joinpath(pkgdir(Metalhead), "test", "fixtures")
const SHA = "0123456789abcdef0123456789abcdef01234567"
const SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

# The files of `repo` at commit SHA in a Hugging Face cache laid out like huggingface_hub's,
# copied from `files` (`path in the repository => local file`).
function cache_repo(cache, repo, files)
    snapshot = joinpath(cache, "models--" * replace(repo, "/" => "--"), "snapshots", SHA)
    for (file, src) in files
        mkpath(dirname(joinpath(snapshot, file)))
        cp(src, joinpath(snapshot, file))
    end
    return snapshot
end

function fixture_files(name)
    dir = joinpath(FIXTURES, name)
    return [relpath(joinpath(root, f), dir) => joinpath(root, f)
            for (root, _, fs) in walkdir(dir) for f in fs if f != "acts.safetensors"]
end

# resnet_basic, with its weights in two shards, without the `num_batches_tracked` scalars
# that SafeTensors.jl can't write and the loader ignores
function sharded_files(dir)
    tensors = SafeTensors.deserialize(joinpath(FIXTURES, "resnet_basic",
        "model.safetensors"))
    names = sort!([k for k in keys(tensors) if ndims(tensors[k]) > 0])
    weight_map = Dict(k => SHARDS[mod1(i, 2)] for (i, k) in enumerate(names))
    for shard in SHARDS
        SafeTensors.serialize(joinpath(dir, shard),
            Dict(k => collect(tensors[k]) for (k, s) in weight_map if s == shard))
    end
    write(joinpath(dir, "index.json"), JSON.json(Dict("weight_map" => weight_map)))
    return ["config.json" => joinpath(FIXTURES, "resnet_basic", "config.json"),
        "model.safetensors.index.json" => joinpath(dir, "index.json"),
        (shard => joinpath(dir, shard) for shard in SHARDS)...]
end

hub(repo; subfolder = "") = ModelSource(repo, nothing, SHA, subfolder)

# The branch `main` of `repo` at commit SHA in the cache.
function cache_main(cache, repo)
    ref = joinpath(cache, "models--" * replace(repo, "/" => "--"), "refs", "main")
    mkpath(dirname(ref))
    return write(ref, SHA)
end

processor_files(repo) = fixture_files(joinpath("processors", replace(repo, "/" => "--")))

# A tokenizer.json for the tokenizer of Qwen3, which strings don't need
function qwen3_files(dir)
    json = Dict("model" => Dict("type" => "BPE", "vocab" => Dict(), "merges" => []),
        "added_tokens" => [], "padding" => nothing, "truncation" => nothing)
    write(joinpath(dir, "tokenizer.json"), JSON.json(json))
    return [processor_files("Qwen/Qwen3-0.6B");
            "tokenizer.json" => joinpath(dir, "tokenizer.json")]
end

function with_cache(f; offline)
    mktempdir() do dir
        cache = joinpath(dir, "hub")
        sharded = cache_repo(cache, "metalhead/sharded", sharded_files(mktempdir(dir)))
        pipeline = cache_repo(cache, "metalhead/pipeline",
            fixture_files("flux2_klein_tiny"))
        # processors of branches
        cache_repo(cache, "metalhead/qwen3", qwen3_files(mktempdir(dir)))
        cache_repo(cache, "metalhead/sam3", processor_files("unsloth/sam3.1"))
        foreach(repo -> cache_main(cache, repo), ["metalhead/qwen3", "metalhead/sam3"])
        withenv("HF_HOME" => dir, "HF_HUB_CACHE" => cache, "HF_TOKEN" => nothing,
            "HF_HUB_OFFLINE" => offline ? "1" : nothing,
            "TRANSFORMERS_OFFLINE" => nothing) do
            return f(sharded, pipeline)
        end
    end
end

# Offline, or without network, the files of a processor that the cache lacks count as
# missing, as do the configurations of `hf_download_first`. A branch resolves to the commit
# that the cache has for it: the files that the processor downloads later come from it,
# without asking the Hub. Without network, a branch warns once, when its first file is read.
function check_processors(; offline)
    T = Metalhead.Transformers
    unreachable = (:warn, r"Cannot reach the Hub, using the cached version")
    load(repo) = offline ? (@test_logs create_processor(repo)) :
                 (@test_logs unreachable create_processor(repo))
    tok = load("metalhead/qwen3")
    @test tok isa T.TokenizersBackend && tok.tokenizer_class == "Qwen2Tokenizer"
    @test tok.tokenizer_file.source.revision == SHA
    path = @test_logs map(_ -> Metalhead.local_path(tok.tokenizer_file), 1:20)
    @test allequal(path) && endswith(only(unique(path)), joinpath(SHA, "tokenizer.json"))
    @test tok([[1, 2], [3]]).input_ids == [2 4; 3 151644]  # padded with <|endoftext|>
    sam = load("metalhead/sam3")
    @test sam isa T.Sam3Processor && sam.tokenizer.tokenizer_file.source.revision == SHA
    # the files that only the cache lacks aren't recorded as missing from the repositories
    @test !any(k -> startswith(k[1], "metalhead/"), Metalhead.MISSING_FILES)
end

@testset "offline" begin
    with_cache(; offline = true) do sharded, pipeline
        @test source_weights(hub("metalhead/sharded")) == joinpath.(sharded, SHARDS)
        model = create_model("metalhead/sharded"; pretrained = true, revision = SHA)
        expected = create_model(joinpath(FIXTURES, "resnet_basic"); pretrained = true)
        x = rand(Float32, 32, 32, 3, 1)
        @test model(x) ≈ expected(x)

        @test source_config(hub("metalhead/pipeline"))["_class_name"] ==
              "Flux2KleinPipeline"
        @test source_weights(hub("metalhead/pipeline"; subfolder = "vae"),
            "diffusion_pytorch_model") ==
              [joinpath(pipeline, "vae", "diffusion_pytorch_model.safetensors")]
        pipe = create_model("metalhead/pipeline"; pretrained = true, revision = SHA)
        expected = create_model(joinpath(FIXTURES, "flux2_klein_tiny"); pretrained = true)
        @test Flux.trainables(pipe) == Flux.trainables(expected)

        @test_throws ["metalhead/absent@$SHA has no safetensors weights in the cache",
            "tried model.safetensors and model.safetensors.index.json",
            "HF_HUB_OFFLINE is set"] source_weights(hub("metalhead/absent"))
        @test_throws ["has no configuration in the cache",
            "tried vae/config.json and vae/model_index.json"] source_config(
            hub("metalhead/absent"; subfolder = "vae"))

        check_processors(; offline = true)

        # a cached index whose shard is missing, or that doesn't parse, is an error
        rm(joinpath(sharded, SHARDS[2]))
        @test_throws "Cannot find metalhead/sharded/$(SHARDS[2])" source_weights(
            hub("metalhead/sharded"))
        write(joinpath(sharded, "model.safetensors.index.json"), "{")
        @test_throws ArgumentError source_weights(hub("metalhead/sharded"))
    end
end

@testset "no network" begin
    with_cache(; offline = false) do sharded, pipeline
        HF.with_endpoint("http://127.0.0.1:1") do
            @test (@test_logs source_weights(hub("metalhead/sharded"))) ==
                  joinpath.(sharded, SHARDS)
            @test (@test_logs source_config(hub("metalhead/pipeline")))["_class_name"] ==
                  "Flux2KleinPipeline"
            @test_throws ["has no configuration in the cache",
                "tried config.json and model_index.json",
                "the Hub can't be reached"] source_config(hub("metalhead/absent"))
            check_processors(; offline = false)
        end
    end
end

parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false")) && @testset "Hub" begin
    repo, sha = first(Metalhead.lookup("resnet18").references)
    @test first(hf_download_first(repo, ["absent.json", "config.json"], "configuration";
        revision = sha)) == "config.json"
    @test_throws "has no configuration (tried absent.json)" hf_download_first(repo,
        ["absent.json"], "configuration"; revision = sha)
    @test_throws "RepoNotFound" hf_download_first("metalhead/absent", ["config.json"],
        "configuration")
end
