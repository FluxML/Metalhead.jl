using Metalhead, Test
using Flux: Flux
include("../parity_setup.jl")
using .ParitySetup

const QWEN3_5_MOE_FIXTURES = ["qwen3_5_moe_text_tiny", "qwen3_5_moe_tiny"]
const T = Metalhead.Transformers

@testset "fixtures" begin
    for name in QWEN3_5_MOE_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    model = fixture_model(fixture("qwen3_5_moe_text_tiny"); eltype = Float32)
    @test check_gradient(model, rand(1:64, 5, 2))
    # through the state carried from a chunk of 64 tokens to the next
    @test check_gradient(model, rand(1:64, 70, 2))
    model = fixture_model(fixture("qwen3_5_moe_tiny"); eltype = Float32)
    acts = Metalhead.Checkpoint(joinpath(fixture("qwen3_5_moe_tiny"), "acts.safetensors"))
    (ids,), inputs = fixture_inputs(acts)
    @test check_gradient(model, ids; pixel_values = Float32.(inputs.pixel_values),
        inputs.image_grid_thw, inputs.attention_mask)
end

@testset "half precision" begin
    # A few tokens go to other experts in Float16, which these random models amplify:
    # PyTorch's float16 errs by 6.3% and 1.7% here (its bfloat16 by 30% and 19%), Float16
    # on the CPU, which accumulates in 16 bits, by 12% and 5.3%.
    for name in QWEN3_5_MOE_FIXTURES
        @test check_half_precision(name; rtol = 0.15)
    end
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, num_hidden_layers = 4,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 8,
        linear_num_key_heads = 1, linear_num_value_heads = 2, linear_key_head_dim = 4,
        linear_value_head_dim = 4, moe_intermediate_size = 6,
        shared_expert_intermediate_size = 5, num_experts = 4, num_experts_per_tok = 2)
    qwen3_5_moe(; kwargs...) = create_model("Qwen3_5MoeForCausalLM"; tiny..., kwargs...)
    families = list_models(; byfamily = true)["qwen"]
    @test "Qwen3_5MoeForCausalLM" in families
    @test "Qwen3_5MoeForConditionalGeneration" in families
    @test Metalhead.lookup("Qwen3_5MoeForCausalLM").ignore == [r"^mtp\."]
    model = qwen3_5_moe()
    @test model isa T.Qwen3_5ForCausalLM
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 70))) == (16, 70)
    L, F = "linear_attention", "full_attention"
    layer_types(m) = [l.self_attn === nothing ? L : F for l in m.model.layers]
    @test layer_types(model) == [L, L, L, F]
    @test layer_types(qwen3_5_moe(; layer_types = [F, L, F, L])) == [F, L, F, L]
    mlp = model.model.layers[1].mlp
    @test mlp isa T.Qwen3_5MoeSparseMoeBlock
    @test size(mlp.gate.weight) == (4, 8)
    @test mlp.gate.norm_topk_prob && mlp.experts.stacked
    @test size(mlp.experts.gate_up_proj) == (8, 12, 4)
    @test size(mlp.shared_expert.gate_proj.weight) == (5, 8)
    @test size(mlp.shared_expert_gate.weight) == (1, 8)
    keys = [p.key for p in first(Metalhead.torch_params(model))]
    @test "model.layers.0.mlp.experts.gate_up_proj" in keys
    @test "model.layers.0.mlp.shared_expert_gate.weight" in keys
    @test model.lm_head isa Flux.Dense
    tied = qwen3_5_moe(; tie_word_embeddings = true)
    @test tied.lm_head.weight === tied.model.embed_tokens.weight
    @test_throws "num_experts_per_tok" qwen3_5_moe(; num_experts_per_tok = 5)
    @test_throws "hidden_act" qwen3_5_moe(; hidden_act = "relu")

    # padding on the right, across chunks of 64 tokens, leaves the other tokens unchanged
    n = 70
    ids = rand(1:16, n, 2)
    mask = [ones(Int, n) [ones(Int, 66); zeros(Int, 4)]]
    padded = model(ids; attention_mask = mask)
    @test padded[:, :, 1] ≈ model(ids[:, 1])
    @test padded[:, 1:66, 2] ≈ model(ids[1:66, 2])

    # the vision-language model: Qwen3.5's, with this decoder
    vlm = create_model("Qwen3_5MoeForConditionalGeneration"; text_config = tiny,
        vision_config = (; depth = 1, hidden_size = 8, num_heads = 2, intermediate_size = 8,
            patch_size = 2, out_hidden_size = 8))
    @test vlm isa T.Qwen3_5ForConditionalGeneration
    @test vlm.model.language_model.layers[1].mlp isa T.Qwen3_5MoeSparseMoeBlock
    @test size(vlm(rand(1:16, 5, 2))) == (16, 5, 2)
    @test feature_info(vlm)[1].module_name == "model.language_model.layers.0"
end

# Qwen3.6-35B-A3B is 72 GB in bfloat16: its reference holds the outputs of its first 4
# layers (3 of linear attention, 1 of full attention) and of its vision encoder, a model of
# 4.8 billion parameters, whose 11 GB of shards it downloads
TEST_LARGE && @testset "references" begin
    for file in reference_files_of("Qwen3_5MoeForConditionalGeneration")
        model = reference_model(file)
        for r in (check_lm_reference(file; model), check_image_reference(file; model))
            @test r.hidden_states <= 1e-4
            @test r.last_logits <= 1e-4
            @test r.generated
        end
    end
end
