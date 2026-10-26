using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

const QWEN2_FIXTURES = ["qwen2_tiny", "qwen2_options"]

@testset "fixtures" begin
    for name in QWEN2_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in QWEN2_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
end

@testset "half precision" begin
    for name in QWEN2_FIXTURES
        @test check_half_precision(name)
    end
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 1)
    model = create_model("Qwen2ForCausalLM"; tiny...)
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)
    @test "Qwen2ForCausalLM" in list_models(; byfamily = true)["qwen"]
    attn = model.model.layers[1].self_attn
    @test size(attn.k_proj.weight) == (4, 8)
    @test attn.q_proj.bias isa AbstractVector && attn.k_proj.bias isa AbstractVector &&
          attn.v_proj.bias isa AbstractVector
    @test attn.o_proj.bias === false
    @test model.model.layers[1].mlp.gate_proj.bias === false

    @test_throws "layer_types" create_model("Qwen2ForCausalLM"; tiny...,
        use_sliding_window = true, max_window_layers = 1)
    @test_throws "layer_types" create_model("Qwen2ForCausalLM"; tiny...,
        layer_types = ("full_attention", "sliding_attention"))
    @test create_model("Qwen2ForCausalLM"; tiny..., use_sliding_window = true,
        sliding_window = nothing, max_window_layers = 0) isa
          Metalhead.Transformers.LlamaForCausalLM
    @test_throws "rope_type" create_model("Qwen2ForCausalLM"; tiny...,
        rope_scaling = (rope_type = "yarn", factor = 4.0))
    # not a field of `Qwen2Config`, but transformers reads it
    wide = create_model("Qwen2ForCausalLM"; tiny..., head_dim = 6)
    @test size(wide.model.layers[1].self_attn.q_proj.weight) == (12, 8)
    @test size(wide(rand(1:16, 5, 3))) == (16, 5, 3)
    @test_throws "divisible" create_model("Qwen2ForCausalLM"; tiny..., hidden_size = 9)
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files_of("Qwen2ForCausalLM")
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
