using Metalhead, Test
using Metalhead: REGISTRY, lookup, model_args, model_source, source_file, unread_keys, JSON
using Metalhead.Transformers: TRANSFORMERS
using Metalhead.Diffusers: DIFFUSERS, DIFFUSERS_PIPELINE, is_component
include(joinpath(@__DIR__, "..", "data.jl"))
using .TestData: fixture

# A copy of the fixture `name` in a directory that can be edited.
function editable_fixture(f, name)
    return mktempdir() do dir
        copy = joinpath(dir, name)
        cp(fixture(name), copy)
        chmod(copy, 0o755; recursive = true)
        return f(copy)
    end
end

function edit_json(file, pairs...)
    config = JSON.parsefile(file)
    foreach(p -> config[first(p)] = last(p), pairs)
    return open(io -> JSON.print(io, config), file, "w")
end

@testset "keys the model doesn't read" begin
    spec = lookup("LlamaForCausalLM")
    config = Dict("model_type" => "llama", "hidden_size" => 8)

    # the keys of transformers and those of the architecture are ignored without a warning
    for key in ("architectures", "transformers_version", "dtype", "use_cache",
        "tie_word_embeddings", "pad_token_id", "temperature", "pretraining_tp")
        config[key] = 1
    end
    args = @test_logs model_args(TRANSFORMERS, spec, config, (;))
    @test args.hidden_size == 8 && !haskey(args, :pretraining_tp)
    @test unread_keys(TRANSFORMERS, spec, config) == String[]

    # a key of another architecture is not, and each is named once
    config["router_aux_loss_coef"] = 0.01
    config["output_gate_type"] = "swish"
    @test unread_keys(TRANSFORMERS, spec, config) ==
          ["output_gate_type", "router_aux_loss_coef"]
    args = @test_logs (:warn,
        r"configuration of LlamaForCausalLM.*`output_gate_type`, `router_aux_loss_coef`") model_args(
        TRANSFORMERS, spec, config, (;))
    @test args.hidden_size == 8 && !haskey(args, :output_gate_type)
    @test unread_keys(TRANSFORMERS, lookup("GraniteMoeForCausalLM"),
        Dict("router_aux_loss_coef" => 0.01)) == String[]

    # a keyword of the builder is read, whatever the lists say
    @test "tie_word_embeddings" in Metalhead.Transformers.IGNORED_CONFIG_KEYS
    @test model_args(TRANSFORMERS, spec, Dict("tie_word_embeddings" => true), (;)).tie_word_embeddings

    # diffusers drops the keys that start with an underscore
    flux = lookup("Flux2Transformer2DModel")
    @test unread_keys(DIFFUSERS, flux, Dict("_class_name" => "x", "_ignore_files" => [])) ==
          String[]
    @test unread_keys(DIFFUSERS, flux, Dict("_class_name" => "x", "new" => 1)) == ["new"]
    @test unread_keys(TRANSFORMERS, spec, Dict("_class_name" => "x")) == ["_class_name"]
end

@testset "create_model warns" begin
    editable_fixture("llama_tiny") do dir
        @test_logs create_model(dir)
        edit_json(joinpath(dir, "config.json"), "future_setting" => 1)
        model = @test_logs (:warn, r"LlamaForCausalLM.*`future_setting`") create_model(dir)
        @test model isa Metalhead.Transformers.LlamaForCausalLM
    end

    editable_fixture("flux2_klein_tiny") do dir
        @test_logs create_model(dir)
        edit_json(joinpath(dir, "scheduler", "scheduler_config.json"), "new_mode" => true)
        edit_json(joinpath(dir, "transformer", "config.json"), "new_attention" => "x")
        pipe = @test_logs (:warn, r"scheduler_config.json.*Flux2KleinPipeline.*`new_mode`") (
            :warn, r"Flux2Transformer2DModel.*`new_attention`") create_model(dir)
        @test pipe isa Metalhead.Diffusers.Flux2KleinPipeline
    end
end

# The configuration a model of `spec` is built from, in the repository `source`: the
# config.json of the transformers model of a sentence-transformers repository, and the
# settings of a pipeline, not its components.
function reference_config(spec, source)
    config = Metalhead.source_config(source)
    haskey(config, "modules.json") &&
        (config = JSON.parsefile(source_file(source, "config.json")))
    spec.format == DIFFUSERS_PIPELINE &&
        (config = filter(p -> !is_component(last(p)), config))
    return config
end

# The checkpoints that an architecture was validated against have no key that it doesn't read:
# one that warned would hide a key that matters among the warnings of every user.
parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false")) &&
    @testset "reference checkpoints" begin
        for spec in sort!(collect(values(REGISTRY)); by = s -> s.name),
            (repo, _) in spec.references

            spec.format in (TRANSFORMERS, DIFFUSERS, DIFFUSERS_PIPELINE) || continue
            config = reference_config(spec, model_source(repo, nothing))
            @test unread_keys(spec.format, spec, config) == String[]
        end

        # a sentence-transformers model builds its transformers model from the config.json
        repo, _ = first(lookup("SentenceTransformer").references)
        source = model_source(repo, nothing)
        @test unread_keys(TRANSFORMERS, lookup("Qwen3ForCausalLM"),
            JSON.parsefile(source_file(source, "config.json"))) == String[]
    end
