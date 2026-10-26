using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

const GEMMA3_FIXTURES = ["gemma3_tiny", "gemma3_options"]

@testset "fixtures" begin
    for name in GEMMA3_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in GEMMA3_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
end

@testset "half precision" begin
    for name in GEMMA3_FIXTURES
        @test check_half_precision(name)
    end
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 6, num_attention_heads = 2, num_key_value_heads = 1,
        head_dim = 4)
    model = create_model("Gemma3ForCausalLM"; tiny...)
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)
    @test list_models(; byfamily = true)["gemma3"] == ["Gemma3ForCausalLM"]
    @test model.lm_head.weight === model.model.embed_tokens.weight
    @test model.model.embed_tokens.embed_scale == Float32(sqrt(8))
    @test all(iszero, model.model.norm.weight)

    # every sixth layer attends to all the previous tokens, unless `layer_types` says
    # otherwise
    layer_types(m) = [l.self_attn.layer_type for l in m.model.layers]
    @test layer_types(model) == [fill(:sliding_attention, 5); :full_attention]
    attn = model.model.layers[1].self_attn
    @test attn.sliding_window == 4096
    @test attn.scaling == 256^-0.5
    @test model.model.layers[6].self_attn.sliding_window === nothing
    pattern = create_model("Gemma3ForCausalLM"; tiny..., sliding_window_pattern = 2)
    @test layer_types(pattern) == repeat([:sliding_attention, :full_attention], 3)

    # a rotary embedding for each layer type, from `rope_parameters` or the older keys
    rope(; kw...) = create_model("Gemma3ForCausalLM"; tiny..., kw...).model.rotary_emb
    @test rope().full_attention.rope_parameters == (rope_type = "default", rope_theta = 1e6)
    @test rope().sliding_attention.rope_parameters ==
          (rope_type = "default", rope_theta = 1e4)
    old = rope(rope_theta = 5e5, rope_local_base_freq = 100,
        rope_scaling = Dict("rope_type" => "linear", "factor" => 8.0))
    @test old.full_attention.rope_parameters ==
          (rope_type = "linear", rope_theta = 5e5, factor = 8.0)
    @test old.sliding_attention.rope_parameters == (rope_type = "default", rope_theta = 100)
    new = rope(rope_parameters = Dict(
        "full_attention" => Dict("rope_type" => "default", "rope_theta" => 3.0),
        "sliding_attention" => Dict("rope_type" => "default", "rope_theta" => 2.0)))
    @test new.full_attention.rope_parameters.rope_theta == 3
    @test new.sliding_attention.rope_parameters.rope_theta == 2
    @test_throws "rope_type" rope(rope_scaling = Dict("rope_type" => "yarn", "factor" =>
        8.0))

    @test_throws "use_bidirectional_attention" create_model("Gemma3ForCausalLM"; tiny...,
        use_bidirectional_attention = true)
    @test_throws "layer_types" create_model("Gemma3ForCausalLM"; tiny...,
        layer_types = ["full_attention"])
    @test_throws "chunked_attention" create_model("Gemma3ForCausalLM"; tiny...,
        layer_types = fill("chunked_attention", 6))
    @test_throws "sliding_window" create_model("Gemma3ForCausalLM"; tiny...,
        sliding_window = nothing)
    @test_throws "hidden_activation" create_model("Gemma3ForCausalLM"; tiny...,
        hidden_activation = "gelu")
    @test_throws "divisible" create_model("Gemma3ForCausalLM"; tiny..., hidden_size = 9)

    # padding on the right leaves the other tokens unchanged; the last padding positions,
    # whose window holds no token, attend to nothing and get zeros, as in transformers
    model = create_model("Gemma3ForCausalLM"; tiny..., sliding_window = 2)
    ids = rand(1:16, 9, 2)
    mask = [ones(Int, 9) [1, 1, 1, 0, 0, 0, 0, 0, 0]]
    padded = model(ids; attention_mask = mask)
    @test all(isfinite, padded)
    @test padded[:, 1:3, 2] ≈ model(ids[1:3, 2])
    @test padded[:, :, 1] ≈ model(ids[:, 1])
    hidden = forward_intermediates(model, ids; indices = [6], intermediates_only = true,
        attention_mask = mask)
    @test only(hidden)[:, 1:3, 2] ≈
          only(forward_intermediates(model, ids[1:3, 2:2]; indices = [6],
        intermediates_only = true))[:, :, 1]
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files_of("Gemma3ForCausalLM")
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
