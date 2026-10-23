using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

const QWEN3_FIXTURES = ["qwen3_tiny", "qwen3_options"]

@testset "fixtures" begin
    for name in QWEN3_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in QWEN3_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 1,
        head_dim = 4)
    model = create_model("Qwen3ForCausalLM"; tiny...)
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)
    @test list_models(; byfamily = true)["qwen3"] == ["Qwen3ForCausalLM"]
    attn = model.model.layers[1].self_attn
    @test attn isa Metalhead.Transformers.Qwen3Attention
    @test size(attn.q_norm.weight) == size(attn.k_norm.weight) == (4,)
    @test attn.q_proj.bias === false

    # `head_dim` need not divide `hidden_size` (Qwen3-0.6B: 16 heads of 128 features, hidden 1024)
    wide = create_model(
        "Qwen3ForCausalLM"; tiny..., hidden_size = 10, num_attention_heads = 4,
        num_key_value_heads = 2, head_dim = 6)
    @test size(wide.model.layers[1].self_attn.q_proj.weight) == (24, 10)
    @test size(forward_features(wide, rand(1:16, 5, 3))) == (10, 5, 3)

    @test_throws "layer_types" create_model("Qwen3ForCausalLM"; tiny...,
        use_sliding_window = true, max_window_layers = 1)
    @test_throws "rope_type" create_model("Qwen3ForCausalLM"; tiny...,
        rope_parameters = Dict("rope_type" => "yarn", "rope_theta" => 1e6, "factor" => 4.0))
    @test_throws "hidden_act" create_model("Qwen3ForCausalLM"; tiny..., hidden_act = "gelu")
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files_of("Qwen3ForCausalLM")
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
