using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

const LLAMA_FIXTURES = ["llama_tiny", "llama_options"]

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

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 1, num_attention_heads = 2)
    model = create_model("LlamaForCausalLM"; tiny...)
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)
    @test size(forward_features(model, rand(1:16, 5, 3))) == (8, 5, 3)
    @test list_models(; byfamily = true)["llama"] == ["LlamaForCausalLM"]
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
    bf16 = create_model("LlamaForCausalLM"; tiny..., eltype = Flux.BFloat16)
    @test eltype(bf16(rand(1:16, 5))) == Flux.BFloat16

    # config.json has `rope_theta` and `rope_scaling` (transformers 4) or `rope_parameters` (5)
    theta(; kw...) = create_model("LlamaForCausalLM"; tiny..., kw...).model.rotary_emb.rope_theta
    @test theta() == 10_000
    @test theta(rope_theta = 500) == 500
    @test theta(rope_parameters = (rope_type = "default", rope_theta = 300.0)) == 300
    @test theta(rope_theta = 500, rope_parameters = Dict("rope_type" => "default")) == 500
    @test_throws "rope_type" theta(rope_scaling = (rope_type = "llama3", factor = 8.0))

    @test_throws "unsupported keyword" create_model("LlamaForCausalLM"; tiny..., num_layers = 2)
    @test_throws "unsupported keyword" create_model(fixture("llama_tiny"); num_layers = 2)
    @test_throws "hidden_act" create_model("LlamaForCausalLM"; tiny..., hidden_act = "gelu")
    @test_throws "divisible" create_model("LlamaForCausalLM"; tiny..., num_key_value_heads = 3)
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("SmolLM2"; org = "HuggingFaceTB")
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
