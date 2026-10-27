# Ported from transformers.models.qwen3_moe.modeling_qwen3_moe (transformers 5.20,
# Apache-2.0, The Hugging Face Team). Validated against test/fixtures/qwen3_moe_* and
# test/references/Qwen--Qwen3-30B-A3B, its first 4 layers.

# The router of Qwen3 MoE and Qwen3.5 MoE: the softmax of the logits `weight * x` over the
# experts, in at least `Float32` as in transformers, and the `top_k` experts of largest
# probability, whose probabilities weight their outputs, renormalised to sum to 1 with
# `norm_topk_prob`. It returns the experts (top_k, n) of the n tokens and their weights.
@concrete struct Qwen3MoeTopKRouter
    weight
    top_k
    norm_topk_prob
end
Flux.@layer Qwen3MoeTopKRouter

layer_params(m::Qwen3MoeTopKRouter) = [("weight", m.weight, :dense)]

function (m::Qwen3MoeTopKRouter)(x)
    T = eltype(x)
    probs = softmax(promote_type(T, Float32).(m.weight * x); dims = 1)
    weights, index = topk(probs, m.top_k)
    weights = m.norm_topk_prob ? weights ./ sum(weights; dims = 1) : weights
    return index, T.(weights)
end

# The experts, gated MLPs of the stacked weights of transformers, `gate_up_proj`
# (hidden, 2 * intermediate, num_experts) and `down_proj` (intermediate, hidden,
# num_experts). The checkpoints of Qwen3.5 MoE store these (`stacked`), those of Qwen3 MoE
# one MLP per expert, which transformers merges on loading (`conversion_mapping.py`): its
# `gate_proj`, `up_proj` and `down_proj` load into slices of the stacked weights.
@concrete struct Qwen3MoeExperts
    gate_up_proj
    down_proj
    act_fn
    stacked
end
Flux.@layer Qwen3MoeExperts

function layer_params(m::Qwen3MoeExperts)
    gate_up, down = m.gate_up_proj, m.down_proj
    m.stacked && return [("gate_up_proj", gate_up, :plain), ("down_proj", down, :plain)]
    n = size(down, 1)
    mlp(e) = [("$(e - 1).gate_proj.weight", view(gate_up, :, 1:n, e), :plain),
        ("$(e - 1).up_proj.weight", view(gate_up, :, (n + 1):(2n), e), :plain),
        ("$(e - 1).down_proj.weight", view(down, :, :, e), :plain)]
    return reduce(vcat, [mlp(e) for e in axes(down, 3)])
end

function (m::Qwen3MoeExperts)(x, index, weights)
    return moe_experts(m.gate_up_proj, m.down_proj, m.act_fn, x, index, weights)
end

function moe_experts_of(; hidden_size, moe_intermediate_size, num_experts, act, init,
        stacked)
    return Qwen3MoeExperts(init(hidden_size, 2 * moe_intermediate_size, num_experts),
        init(moe_intermediate_size, hidden_size, num_experts), act, stacked)
end

@concrete struct Qwen3MoeSparseMoeBlock
    experts
    gate
end
Flux.@layer Qwen3MoeSparseMoeBlock

function (m::Qwen3MoeSparseMoeBlock)(x)
    tokens = reshape(x, size(x, 1), :)
    index, weights = m.gate(tokens)
    return reshape(m.experts(tokens, index, weights), size(x))
end

function check_experts(; num_experts, num_experts_per_tok)
    1 <= num_experts_per_tok <= num_experts ||
        error("`num_experts_per_tok` ($num_experts_per_tok) must be in 1:$num_experts, `num_experts`")
    return nothing
end

"""
    qwen3_moe_for_causal_lm(; vocab_size = 151936, hidden_size = 2048, num_hidden_layers = 24, kwargs...)

Port of transformers' `Qwen3MoeForCausalLM` (Qwen3-30B-A3B, Qwen3-235B-A22B), built from the
fields of its `Qwen3MoeConfig` with the same names and defaults, and `head_dim`,
`rope_theta` and `rope_scaling`, which transformers reads from config.json. It takes and
returns the arrays of [`llama_for_causal_lm`](@ref
Metalhead.Transformers.llama_for_causal_lm).

Qwen3 MoE is Qwen3 whose MLP is a sparse mixture of `num_experts` gated MLPs of
`moe_intermediate_size` features: each token goes through the `num_experts_per_tok` experts
of largest router probability, weighted by these probabilities, renormalised to sum to 1 if
`norm_topk_prob`. The layers in `mlp_only_layers` (from 0), and all if `num_experts = 0`,
have a dense MLP of `intermediate_size` features instead, and so do those whose number
(from 1) isn't a multiple of `decoder_sparse_step`. config.json files written by
transformers 5 name `num_experts` `num_local_experts`. Sliding-window attention
(`use_sliding_window`) and rotary embeddings other than the default, `"linear"` and
`"llama3"` ones error. The training options (`output_router_logits`,
`router_aux_loss_coef`) are ignored.
"""
function qwen3_moe_for_causal_lm(; vocab_size = 151_936, hidden_size = 2048,
        intermediate_size = 6144, num_hidden_layers = 24, num_attention_heads = 32,
        num_key_value_heads = 4, head_dim = nothing, hidden_act = "silu",
        max_position_embeddings = 32_768, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = false, rope_parameters = nothing, rope_theta = nothing,
        rope_scaling = nothing, attention_bias = false, use_sliding_window = false,
        sliding_window = 4096, attention_dropout = 0.0, decoder_sparse_step = 1,
        moe_intermediate_size = 768, num_experts_per_tok = 8, num_experts = 128,
        num_local_experts = nothing, norm_topk_prob = false, mlp_only_layers = nothing)
    name = "Qwen3MoeForCausalLM"
    use_sliding_window && sliding_window !== nothing &&
        error("sliding-window attention (`use_sliding_window = true`) is not supported by the Metalhead port of $name")
    num_experts = something(num_local_experts, num_experts)
    num_experts == 0 || check_experts(; num_experts, num_experts_per_tok)
    decoder_sparse_step >= 1 ||
        error("`decoder_sparse_step` ($decoder_sparse_step) must be positive")
    dense_layers = something(mlp_only_layers, ())
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    rope = rope_parameters_of(name; rope_theta, rope_scaling, rope_parameters,
        max_position_embeddings)
    act = act_fn(hidden_act)
    init = normal_init(initializer_range)
    attention(dense) = Qwen3Attention(head_dim,
        dense(hidden_size => num_attention_heads * head_dim, attention_bias),
        dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
        dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
        dense(num_attention_heads * head_dim => hidden_size, attention_bias),
        RMSNorm(head_dim; eps = rms_norm_eps), RMSNorm(head_dim; eps = rms_norm_eps),
        dropout_or_identity(attention_dropout))
    function mlp(dense, i)
        if i ∉ dense_layers && num_experts > 0 && (i + 1) % decoder_sparse_step == 0
            router = Qwen3MoeTopKRouter(init(num_experts, hidden_size),
                num_experts_per_tok, norm_topk_prob)
            experts = moe_experts_of(; hidden_size, moe_intermediate_size, num_experts,
                act, init, stacked = false)
            return Qwen3MoeSparseMoeBlock(experts, router)
        end
        return LlamaMLP(dense(hidden_size => intermediate_size, false),
            dense(hidden_size => intermediate_size, false),
            dense(intermediate_size => hidden_size, false), act)
    end
    return llama_like(attention; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, hidden_act, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters = rope, head_dim, mlp)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3MoeForCausalLM", qwen3_moe_for_causal_lm;
    family = "qwen",
    references = ["Qwen/Qwen3-30B-A3B" => "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"]))
