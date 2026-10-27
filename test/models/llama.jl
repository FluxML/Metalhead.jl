using Metalhead, Test
using Flux: BFloat16
include("../parity_setup.jl")
using .ParitySetup

const LLAMA_FIXTURES = ["llama_tiny", "llama_options", "llama_rope"]

@testset "fixtures" begin
    for name in LLAMA_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in LLAMA_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
end

@testset "half precision" begin
    for name in LLAMA_FIXTURES
        @test check_half_precision(name)
    end
    # `create_model(...; eltype)` loads the checkpoint into the Float32 norms
    half = fixture_model(fixture("llama_tiny"); eltype = Float16)
    @test eltype(half.model.norm.weight) == Float32
    @test eltype(half.model.embed_tokens.weight) == Float16
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 1, num_attention_heads = 2)
    model = create_model("LlamaForCausalLM"; tiny...)
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)
    @test size(forward_features(model, rand(1:16, 5, 3))) == (8, 5, 3)
    @test list_models(; byfamily = true)["llama"] == ["LlamaForCausalLM"]

    # the output of a TextProcessor, a vector for one text, a matrix for a batch
    ids = rand(1:16, 5, 2)
    mask = [1 1; 1 1; 1 1; 1 0; 1 0]
    @test model((; input_ids = ids, attention_mask = mask)) ≈ model(ids; attention_mask = mask)
    @test model((; input_ids = ids[:, 1])) ≈ model(ids[:, 1])
    @test forward_features(model, (; input_ids = ids, attention_mask = mask)) ≈
          forward_features(model, ids; attention_mask = mask)
    @test_throws "has no `input_ids` (it has `attention_mask`); pass `text` to the processor" model((;
        attention_mask = mask))
    # the features of one sequence, as those of a batch of one
    features, hidden = forward_intermediates(model, ids[:, 1])
    batched = forward_intermediates(model, ids[:, 1:1])
    @test features ≈ batched[1][:, :, 1] && size(features) == (8, 5)
    @test only(hidden) ≈ only(batched[2])[:, :, 1]
    @test only(forward_intermediates(model, ids[:, 1]; attention_mask = ones(Int, 5),
        intermediates_only = true)) ≈ only(hidden)
    @test forward_intermediates(model, (; input_ids = ids, attention_mask = mask);
        intermediates_only = true) ≈
          forward_intermediates(model, ids; attention_mask = mask, intermediates_only = true)
    @test feature_info(model) == [(; num_chs = 8, reduction = 1, module_name = "model.layers.0")]
    @test length(feature_info(create_model("LlamaForCausalLM"; tiny...,
        num_hidden_layers = 3))) == 3
    config = Metalhead.JSON.parsefile(joinpath(fixture("llama_tiny"), "config.json"))
    @test Metalhead.identify(config) ==
          (Metalhead.Transformers.TRANSFORMERS, "LlamaForCausalLM")
    @test_throws "several architectures" Metalhead.identify(merge(config,
        Dict("architectures" => ["LlamaForCausalLM", "LlamaModel"])))

    tied = create_model("LlamaForCausalLM"; tiny..., tie_word_embeddings = true)
    @test tied.lm_head.weight === tied.model.embed_tokens.weight
    @test last(Metalhead.torch_params(tied)) == ["lm_head.weight" => "model.embed_tokens.weight"]
    half = create_model("LlamaForCausalLM"; tiny..., tie_word_embeddings = true,
        eltype = Float16)
    @test half.lm_head.weight === half.model.embed_tokens.weight
    @test eltype(half(rand(1:16, 5))) == Float16
    # BFloat16 is only run on GPUs, not tested on the CPU
    bf16 = create_model("LlamaForCausalLM"; tiny..., eltype = BFloat16)
    @test eltype(bf16.model.norm.weight) == Float32
    @test eltype(bf16.model.embed_tokens.weight) == BFloat16

    # config.json has `rope_theta` and `rope_scaling` (transformers 4) or `rope_parameters` (5)
    rotary(; kw...) = create_model("LlamaForCausalLM"; tiny..., kw...).model.rotary_emb
    rope(; kw...) = rotary(; kw...).rope_parameters
    @test rope() == (rope_type = "default", rope_theta = 10_000)
    @test rope(rope_theta = 500).rope_theta == 500
    @test rope(rope_parameters = (rope_type = "default", rope_theta = 300.0)).rope_theta == 300
    @test rope(rope_theta = 500,
        rope_parameters = Dict("rope_type" => "default")).rope_theta == 500
    # Llama 3.1 and later, whose `original_max_position_embeddings` defaults to
    # `max_position_embeddings`
    llama3 = Dict("rope_type" => "llama3", "factor" => 32.0, "low_freq_factor" => 1.0,
        "high_freq_factor" => 4.0)
    @test rope(rope_scaling = llama3, rope_theta = 5e5, max_position_embeddings = 1000) ==
          (rope_type = "llama3", rope_theta = 5e5, factor = 32.0, low_freq_factor = 1.0,
        high_freq_factor = 4.0, original_max_position_embeddings = 1000)
    @test rope(rope_scaling = (type = "linear", factor = 2.0)) ==
          (rope_type = "linear", rope_theta = 10_000, factor = 2.0)
    @test_throws "rope_type" rope(rope_scaling = (rope_type = "yarn", factor = 8.0))
    @test_throws "partial_rotary_factor" rope(
        rope_scaling = (rope_type = "linear", factor = 2.0, partial_rotary_factor = 0.5))

    @test_throws "unsupported keyword" create_model("LlamaForCausalLM"; tiny..., num_layers = 2)
    @test_throws "unsupported keyword" create_model(fixture("llama_tiny"); num_layers = 2)
    @test_throws "hidden_act" create_model("LlamaForCausalLM"; tiny..., hidden_act = "gelu")
    @test_throws "divisible" create_model("LlamaForCausalLM"; tiny..., num_key_value_heads = 3)
    @test_throws "divisible" create_model("LlamaForCausalLM"; tiny..., hidden_size = 9,
        head_dim = 4)
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files_of("LlamaForCausalLM")
        # Llama 3.2 1B downloads 2.5 GB
        contains(file, "Llama-3.2") && !TEST_LARGE && continue
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
