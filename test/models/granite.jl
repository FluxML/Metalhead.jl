using Metalhead, Test
using Flux: swish
include("../parity_setup.jl")
using .ParitySetup

const GRANITE_FIXTURES = ["granite_tiny", "granite_options", "granitemoe_tiny",
    "granitemoe_options", "granitemoeshared_tiny", "granitemoehybrid_tiny",
    "granitemoehybrid_options"]
const GRANITE_ARCHS = ["GraniteForCausalLM", "GraniteMoeForCausalLM",
    "GraniteMoeSharedForCausalLM", "GraniteMoeHybridForCausalLM"]

@testset "fixtures" begin
    for name in GRANITE_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in GRANITE_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
end

@testset "half precision" begin
    for name in GRANITE_FIXTURES
        @test check_half_precision(name)
    end
end

# the output of a tokenizer, padded on either side
foreach(check_tokenizer_inputs,
    ["granite_tiny", "granitemoe_tiny", "granitemoeshared_tiny", "granitemoehybrid_tiny"])

@testset "experts" begin
    # each token goes through its experts, weighted; expert 4 gets no token
    hidden, intermediate, num_experts, n = 5, 3, 4, 6
    gate_up = randn(hidden, 2 * intermediate, num_experts)
    down = randn(intermediate, hidden, num_experts)
    x = randn(hidden, n)
    index = [1 3 2 2 1 3; 2 1 3 1 3 2]
    weights = rand(2, n)
    y = Metalhead.Transformers.moe_experts(gate_up, down, swish, x, index, weights)
    function expert(e, v)
        h = gate_up[:, :, e]' * v
        return down[:, :, e]' * (swish.(h[1:intermediate]) .* h[(intermediate + 1):end])
    end
    @test y ≈ reduce(hcat,
        [sum(weights[j, t] * expert(index[j, t], x[:, t]) for j in 1:2) for t in 1:n])
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 6,
        num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 1)
    moe = (; num_local_experts = 4, num_experts_per_tok = 2)
    hybrid = (; layer_types = ["attention", "attention"], moe...)
    @test list_models(; byfamily = true)["granite"] == sort(GRANITE_ARCHS)
    for (arch, kwargs) in zip(GRANITE_ARCHS, [(;), moe, moe, hybrid])
        model = create_model(arch; tiny..., kwargs...)
        @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
        @test size(model(rand(1:16, 5))) == (16, 5)
    end

    # the multipliers, and the head size of transformers, `hidden_size ÷ num_attention_heads`
    model = create_model("GraniteForCausalLM"; tiny..., hidden_size = 9,
        embedding_multiplier = 12, attention_multiplier = 0.25, residual_multiplier = 0.5,
        logits_scaling = 4, tie_word_embeddings = true)
    @test model.model.embedding_multiplier === 12.0
    @test model.logits_scaling === 4.0
    attn = model.model.layers[1].self_attn
    @test attn.scaling === 0.25 && attn.head_dim == 4
    @test model.model.layers[1].residual_multiplier === 0.5
    @test model.lm_head.weight === model.model.embed_tokens.weight
    x = forward_features(model, rand(1:16, 5))
    @test forward_head(model, x) ≈ model.lm_head(x) ./ 4

    @test_throws "num_experts_per_tok" create_model("GraniteMoeForCausalLM"; tiny...,
        num_local_experts = 2, num_experts_per_tok = 3)
    @test_throws "num_local_experts" create_model("GraniteMoeForCausalLM"; tiny...,
        num_local_experts = 0)
    @test_throws "hidden_act" create_model("GraniteMoeForCausalLM"; tiny..., moe...,
        hidden_act = "relu")
    shared(n) = create_model("GraniteMoeSharedForCausalLM"; tiny..., moe...,
        shared_intermediate_size = n).model.layers[1].shared_mlp
    @test shared(0) === nothing
    @test size(shared(5).input_linear.weight) == (10, 8)

    # Granite 4.0 without Mamba layers, with or without experts and rotary embedding
    granite4(; kw...) = create_model("GraniteMoeHybridForCausalLM"; tiny..., hybrid..., kw...)
    @test granite4().model.rotary_emb === nothing
    @test granite4(position_embedding_type = "rope").model.rotary_emb isa
          Metalhead.Transformers.RotaryEmbedding
    layer = granite4(num_local_experts = 0, num_experts_per_tok = 0).model.layers[1]
    @test layer.block_sparse_moe === nothing
    @test size(layer.shared_mlp.input_linear.weight) == (2048, 8)
    @test granite4(layer_types = ["full_attention", "attention"]) isa
          Metalhead.Transformers.GraniteForCausalLM
    @test_throws "Mamba" granite4(layer_types = nothing)
    @test_throws "Mamba" granite4(layer_types = ["attention", "mamba"])
    @test_throws "sliding_attention" granite4(layer_types = ["attention",
        "sliding_attention"])
    @test_throws "layer_types" granite4(layer_types = ["attention"])

    # padding on the right leaves the other tokens unchanged, experts included
    model = create_model("GraniteMoeSharedForCausalLM"; tiny..., moe...,
        shared_intermediate_size = 4)
    ids = rand(1:16, 7, 2)
    padded = model(ids; attention_mask = [ones(Int, 7) [1, 1, 1, 0, 0, 0, 0]])
    @test padded[:, :, 1] ≈ model(ids[:, 1])
    @test padded[:, 1:3, 2] ≈ model(ids[1:3, 2])
end

TEST_NETWORK && @testset "references" begin
    for arch in GRANITE_ARCHS, file in reference_files_of(arch)
        # granite-4.0-350m downloads 0.7 GB, the others 2.7 to 13 GB
        arch == "GraniteMoeHybridForCausalLM" || TEST_LARGE || continue
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
