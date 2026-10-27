# Ported from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe (transformers 5.20,
# Apache-2.0, The Hugging Face Team): Qwen3.5 with a sparse mixture of experts and a shared
# expert in every layer. Validated against test/fixtures/qwen3_5_moe_* and
# test/references/Qwen--Qwen3.6-35B-A3B, its first 4 layers.
#
# The models are those of qwen3_5.jl with the MLP `Qwen3_5MoeSparseMoeBlock`. The
# checkpoints of Qwen (Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, Qwen3.8-2.4T-A95B) store the
# experts stacked, as transformers' parameters `experts.gate_up_proj` and
# `experts.down_proj`; `save_pretrained` writes one MLP per expert instead, which this port
# doesn't load.

# The experts of the router `gate`, and the `shared_expert`, a gated MLP that every token
# goes through, weighted by the sigmoid of `shared_expert_gate`.
@concrete struct Qwen3_5MoeSparseMoeBlock
    gate
    experts
    shared_expert
    shared_expert_gate
end
Flux.@layer Qwen3_5MoeSparseMoeBlock

function (m::Qwen3_5MoeSparseMoeBlock)(x)
    tokens = reshape(x, size(x, 1), :)
    index, weights = m.gate(tokens)
    shared = sigmoid.(m.shared_expert_gate(tokens)) .* m.shared_expert(tokens)
    return reshape(m.experts(tokens, index, weights) .+ shared, size(x))
end

"""
    qwen3_5_moe_text_model(; vocab_size = 248320, hidden_size = 2048, num_hidden_layers = 40, kwargs...)

Port of transformers' `Qwen3_5MoeTextModel`, the decoder of Qwen3.5 MoE, built from the
fields of its `Qwen3_5MoeTextConfig` with the same names and defaults, and
`full_attention_interval`, `rope_theta`, `rope_scaling` and `partial_rotary_factor`, which
the config reads from config.json. It is the decoder of [`qwen3_5_text_model`](@ref
Metalhead.Transformers.qwen3_5_text_model), with its inputs and outputs, whose MLPs are
sparse mixtures of `num_experts` gated MLPs of `moe_intermediate_size` features with a
shared expert: each token goes through the `num_experts_per_tok` experts of largest router
probability, weighted by these probabilities renormalised to sum to 1, and through the
shared expert, a gated MLP of `shared_expert_intermediate_size` features weighted by a
sigmoid gate. The training options (`output_router_logits`, `router_aux_loss_coef`) are
ignored.
"""
function qwen3_5_moe_text_model(; vocab_size = 248_320, hidden_size = 2048,
        num_hidden_layers = 40, num_attention_heads = 16, num_key_value_heads = 2,
        hidden_act = "silu", max_position_embeddings = 32_768, initializer_range = 0.02,
        rms_norm_eps = 1e-6, rope_parameters = nothing, rope_theta = nothing,
        rope_scaling = nothing, partial_rotary_factor = nothing, attention_bias = false,
        attention_dropout = 0.0, head_dim = 256, linear_conv_kernel_dim = 4,
        linear_key_head_dim = 128, linear_value_head_dim = 128, linear_num_key_heads = 16,
        linear_num_value_heads = 32, moe_intermediate_size = 512,
        shared_expert_intermediate_size = 512, num_experts_per_tok = 8, num_experts = 256,
        layer_types = nothing, full_attention_interval = 4)
    check_experts(; num_experts, num_experts_per_tok)
    init = normal_init(initializer_range)
    function mlp(dense, act)
        shared = shared_expert_intermediate_size
        return Qwen3_5MoeSparseMoeBlock(
            Qwen3MoeTopKRouter(init(num_experts, hidden_size), num_experts_per_tok, true),
            moe_experts_of(; hidden_size, moe_intermediate_size, num_experts, act, init,
                stacked = true),
            LlamaMLP(dense(hidden_size => shared), dense(hidden_size => shared),
                dense(shared => hidden_size), act),  # Qwen3_5MoeMLP
            dense(hidden_size => 1))
    end
    return qwen3_5_text_like(mlp; vocab_size, hidden_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, hidden_act, initializer_range,
        rms_norm_eps, rope_parameters, rope_theta, rope_scaling, partial_rotary_factor,
        attention_bias, attention_dropout, head_dim, linear_conv_kernel_dim,
        linear_key_head_dim, linear_value_head_dim, linear_num_key_heads,
        linear_num_value_heads, layer_types, full_attention_interval)
end

"""
    qwen3_5_moe_for_causal_lm(; vocab_size = 248320, hidden_size = 2048, num_hidden_layers = 40, kwargs...)

Port of transformers' `Qwen3_5MoeForCausalLM`, the text model of Qwen3.5 MoE
(Qwen3.8-2.4T-A95B): the decoder of [`qwen3_5_moe_text_model`](@ref
Metalhead.Transformers.qwen3_5_moe_text_model), with its keywords, and the output projection
`lm_head`, tied to the token embeddings with `tie_word_embeddings`. It takes and returns the
arrays of [`qwen3_5_for_causal_lm`](@ref Metalhead.Transformers.qwen3_5_for_causal_lm). The
multi-token prediction weights (`mtp.*`) of the checkpoints are not loaded, as in
transformers.
"""
function qwen3_5_moe_for_causal_lm(; vocab_size = 248_320, hidden_size = 2048,
        num_hidden_layers = 40, num_attention_heads = 16, num_key_value_heads = 2,
        hidden_act = "silu", max_position_embeddings = 32_768, initializer_range = 0.02,
        rms_norm_eps = 1e-6, tie_word_embeddings = false, rope_parameters = nothing,
        rope_theta = nothing, rope_scaling = nothing, partial_rotary_factor = nothing,
        attention_bias = false, attention_dropout = 0.0, head_dim = 256,
        linear_conv_kernel_dim = 4, linear_key_head_dim = 128, linear_value_head_dim = 128,
        linear_num_key_heads = 16, linear_num_value_heads = 32, moe_intermediate_size = 512,
        shared_expert_intermediate_size = 512, num_experts_per_tok = 8, num_experts = 256,
        layer_types = nothing, full_attention_interval = 4)
    model = qwen3_5_moe_text_model(; vocab_size, hidden_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, hidden_act, max_position_embeddings,
        initializer_range, rms_norm_eps, rope_parameters, rope_theta, rope_scaling,
        partial_rotary_factor, attention_bias, attention_dropout, head_dim,
        linear_conv_kernel_dim, linear_key_head_dim, linear_value_head_dim,
        linear_num_key_heads, linear_num_value_heads, moe_intermediate_size,
        shared_expert_intermediate_size, num_experts_per_tok, num_experts, layer_types,
        full_attention_interval)
    return qwen3_5_causal_lm(model; tie_word_embeddings, initializer_range)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3_5MoeForCausalLM", qwen3_5_moe_for_causal_lm;
    family = "qwen", ignore = [r"^mtp\."]))

"""
    qwen3_5_moe_for_conditional_generation(; text_config, vision_config, image_token_id = 248056,
                                           tie_word_embeddings = false)

Port of transformers' `Qwen3_5MoeForConditionalGeneration`, the vision-language model of the
Qwen3.5 MoE and Qwen3.6 MoE checkpoints (Qwen3.6-35B-A3B), built from its
`Qwen3_5MoeConfig`:
the model of [`qwen3_5_for_conditional_generation`](@ref
Metalhead.Transformers.qwen3_5_for_conditional_generation), with its inputs, outputs and
processor, whose decoder is [`qwen3_5_moe_text_model`](@ref
Metalhead.Transformers.qwen3_5_moe_text_model) of `text_config`. Videos and the multi-token
prediction weights (`mtp.*`) of the checkpoints are not supported.
"""
function qwen3_5_moe_for_conditional_generation(; text_config = (;), vision_config = (;),
        image_token_id = 248_056, tie_word_embeddings = false)
    return qwen3_5_vision_language(qwen3_5_moe_text_model; text_config, vision_config,
        image_token_id, tie_word_embeddings)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3_5MoeForConditionalGeneration",
    qwen3_5_moe_for_conditional_generation;
    family = "qwen", ignore = [r"^mtp\."],
    references = ["Qwen/Qwen3.6-35B-A3B" => "995ad96eacd98c81ed38be0c5b274b04031597b0"]))
