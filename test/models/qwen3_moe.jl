using Metalhead, Test
using Flux: Flux
include("../parity_setup.jl")
using .ParitySetup

const QWEN3_MOE_FIXTURES = ["qwen3_moe_tiny", "qwen3_moe_options"]
const T = Metalhead.Transformers

@testset "fixtures" begin
    for name in QWEN3_MOE_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in QWEN3_MOE_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
end

@testset "half precision" begin
    for name in QWEN3_MOE_FIXTURES
        @test check_half_precision(name)
    end
end

@testset "router and experts" begin
    # the softmax over all the experts, then the top 2, renormalised or not
    weight = [1.0 0.0; 0.0 1.0; 0.5 0.5]  # (num_experts, hidden)
    x = [1.0 0.0; 3.0 -1.0]
    logits = weight * x  # [1 0; 3 -1; 2 -0.5]
    probs = exp.(logits) ./ sum(exp.(logits); dims = 1)
    index, weights = T.Qwen3MoeTopKRouter(weight, 2, false)(x)
    @test index == [2 1; 3 3]
    @test weights ≈ [probs[2, 1] probs[1, 2]; probs[3, 1] probs[3, 2]]
    _, normed = T.Qwen3MoeTopKRouter(weight, 2, true)(x)
    @test normed ≈ weights ./ sum(weights; dims = 1)
    @test eltype(last(T.Qwen3MoeTopKRouter(Float16.(weight), 2, true)(Float16.(x)))) ==
          Float16

    # the experts' weights of a checkpoint with an MLP per expert, or stacked
    experts = T.Qwen3MoeExperts(zeros(4, 6, 2), zeros(3, 4, 2), Flux.swish, false)
    keys = [p.key for p in first(Metalhead.torch_params(experts))]
    @test keys == ["0.gate_proj.weight", "0.up_proj.weight", "0.down_proj.weight",
        "1.gate_proj.weight", "1.up_proj.weight", "1.down_proj.weight"]
    shapes = ["gate_proj" => (3, 4), "up_proj" => (3, 4), "down_proj" => (4, 3)]  # torch's
    w = Dict("$e.$p.weight" => randn(size...) for e in 0:1 for (p, size) in shapes)
    mktempdir() do dir
        file = joinpath(dir, "experts.safetensors")
        SafeTensors.serialize(file, w)
        Metalhead.load_state!(experts, Metalhead.Checkpoint(file))
    end
    # a torch weight (out, in) is a slice (in, out) of the stacked weights
    @test experts.gate_up_proj[:, 1:3, 2] == permutedims(w["1.gate_proj.weight"])
    @test experts.gate_up_proj[:, 4:6, 1] == permutedims(w["0.up_proj.weight"])
    @test experts.down_proj[:, :, 2] == permutedims(w["1.down_proj.weight"])
    stacked = T.Qwen3MoeExperts(zeros(4, 6, 2), zeros(3, 4, 2), Flux.swish, true)
    @test [p.key for p in first(Metalhead.torch_params(stacked))] ==
          ["gate_up_proj", "down_proj"]
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1,
        head_dim = 4, moe_intermediate_size = 6, num_experts = 4, num_experts_per_tok = 2)
    qwen3_moe(; kwargs...) = create_model("Qwen3MoeForCausalLM"; tiny..., kwargs...)
    @test "Qwen3MoeForCausalLM" in list_models(; byfamily = true)["qwen"]
    model = qwen3_moe()
    @test model isa T.LlamaForCausalLM
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)
    layer = model.model.layers[1]
    @test layer.self_attn isa T.Qwen3Attention
    @test layer.mlp isa T.Qwen3MoeSparseMoeBlock
    @test size(layer.mlp.gate.weight) == (4, 8)
    @test size(layer.mlp.experts.gate_up_proj) == (8, 12, 4)
    @test size(layer.mlp.experts.down_proj) == (6, 8, 4)
    @test !layer.mlp.gate.norm_topk_prob && !layer.mlp.experts.stacked

    # sparse layers: every `decoder_sparse_step`-th, but those of `mlp_only_layers`, and
    # none without experts
    sparse(m) = [l.mlp isa T.Qwen3MoeSparseMoeBlock for l in m.model.layers]
    @test sparse(model) == [true, true, true, true]
    @test sparse(qwen3_moe(; decoder_sparse_step = 2)) == [false, true, false, true]
    @test sparse(qwen3_moe(; mlp_only_layers = [0, 2])) == [false, true, false, true]
    dense = qwen3_moe(; num_experts = 0)
    @test sparse(dense) == [false, false, false, false]
    @test size(dense.model.layers[1].mlp.gate_proj.weight) == (12, 8)
    # transformers 5 writes `num_experts` as `num_local_experts`
    @test size(qwen3_moe(; num_local_experts = 3).model.layers[1].mlp.gate.weight) == (3, 8)
    @test qwen3_moe(; norm_topk_prob = true).model.layers[1].mlp.gate.norm_topk_prob
    # `head_dim` defaults to `hidden_size ÷ num_attention_heads`
    @test create_model("Qwen3MoeForCausalLM"; tiny..., head_dim = nothing,
        num_attention_heads = 4).model.layers[1].self_attn.head_dim == 2
    tied = qwen3_moe(; tie_word_embeddings = true)
    @test tied.lm_head.weight === tied.model.embed_tokens.weight
    # `sliding_window` is ignored without `use_sliding_window`, as in transformers
    @test qwen3_moe(; sliding_window = 2) isa T.LlamaForCausalLM
    @test qwen3_moe(; use_sliding_window = true, sliding_window = nothing) isa
          T.LlamaForCausalLM

    @test_throws "use_sliding_window" qwen3_moe(; use_sliding_window = true)
    @test_throws "num_experts_per_tok" qwen3_moe(; num_experts_per_tok = 5)
    @test_throws "decoder_sparse_step" qwen3_moe(; decoder_sparse_step = 0)
    @test_throws "rope_type" qwen3_moe(;
        rope_parameters = Dict("rope_type" => "yarn", "rope_theta" => 1e6, "factor" => 4.0))
    @test_throws "hidden_act" qwen3_moe(; hidden_act = "relu")

    # padding on the right leaves the other tokens unchanged, experts included
    ids = rand(1:16, 7, 2)
    padded = model(ids; attention_mask = [ones(Int, 7) [1, 1, 1, 0, 0, 0, 0]])
    @test padded[:, :, 1] ≈ model(ids[:, 1])
    @test padded[:, 1:3, 2] ≈ model(ids[1:3, 2])
    last_layer = forward_intermediates(model, ids; indices = [4], intermediates_only = true)
    @test only(last_layer) ≈ last(forward_intermediates(model, ids)[2])
end

# Qwen3-30B-A3B is 61 GB in bfloat16: its reference holds the outputs of its first 4 layers
# (a model of 3.1 billion parameters, whose 9 GB of shards it downloads)
TEST_LARGE && @testset "references" begin
    for file in reference_files_of("Qwen3MoeForCausalLM")
        r = check_lm_reference(file)
        @test r.hidden_states <= 1e-4
        @test r.last_logits <= 1e-4
        @test r.generated
    end
end
