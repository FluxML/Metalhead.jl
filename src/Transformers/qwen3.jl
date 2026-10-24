# Ported from transformers.models.qwen3.modeling_qwen3 (transformers 5.20, Apache-2.0,
# The Hugging Face Team). Validated against test/fixtures/qwen3_* and
# test/references/Qwen--Qwen3-0.6B.

# `LlamaAttention` with an RMSNorm over the features of each query and key head, before the
# rotary embedding.
@concrete struct Qwen3Attention
    head_dim
    q_proj
    k_proj
    v_proj
    o_proj
    q_norm
    k_norm
    attention_dropout
end
Flux.@layer Qwen3Attention

function (m::Qwen3Attention)(x, position_embeddings, kv_lengths = nothing)
    _, n, b = size(x)
    cos, sin = position_embeddings
    q = m.q_norm(reshape(m.q_proj(x), m.head_dim, :, n, b))
    k = m.k_norm(reshape(m.k_proj(x), m.head_dim, :, n, b))
    v = reshape(m.v_proj(x), m.head_dim, :, n, b)
    q = NNlib.apply_rotary_embedding(q, cos, sin)
    k = NNlib.apply_rotary_embedding(k, cos, sin)
    y = NNlib.scaled_dot_product_attention(q, k, v;
        fdrop = attention_fdrop(m.attention_dropout, q),
        is_causal = true, kv_lengths)
    return m.o_proj(reshape(y, :, n, b))
end

"""
    qwen3_for_causal_lm(; vocab_size = 151936, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `Qwen3ForCausalLM`, built from the fields of its `Qwen3Config` with the
same names and defaults. transformers derives Qwen3 from Llama's classes, with a
`Qwen3Attention` that normalises each query and key head, so the model is a
`LlamaForCausalLM` (see
[`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm) for its inputs and
outputs). Sliding-window attention (`use_sliding_window`, `layer_types`) and rotary
embeddings other than the default, `"linear"` and `"llama3"` ones error.
"""
function qwen3_for_causal_lm(; vocab_size = 151936, hidden_size = 4096,
        intermediate_size = 22016, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = 32, head_dim = 128, hidden_act = "silu",
        max_position_embeddings = 32768, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = false, rope_theta = nothing, rope_scaling = nothing,
        rope_parameters = nothing, attention_bias = false, use_sliding_window = false,
        sliding_window = 4096, max_window_layers = 28, layer_types = nothing,
        attention_dropout = 0.0)
    check_full_attention("Qwen3ForCausalLM"; num_hidden_layers, use_sliding_window,
        sliding_window, max_window_layers, layer_types)
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    rope = rope_parameters_of(
        "Qwen3ForCausalLM"; rope_theta, rope_scaling, rope_parameters,
        max_position_embeddings)
    attention(dense) = Qwen3Attention(head_dim,
        dense(hidden_size => num_attention_heads * head_dim, attention_bias),
        dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
        dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
        dense(num_attention_heads * head_dim => hidden_size, attention_bias),
        RMSNorm(head_dim; eps = rms_norm_eps), RMSNorm(head_dim; eps = rms_norm_eps),
        dropout_or_identity(attention_dropout))
    return llama_like(attention; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, hidden_act, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters = rope, head_dim)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3ForCausalLM", qwen3_for_causal_lm;
    family = "qwen",
    references = ["Qwen/Qwen3-0.6B" => "c1899de289a04d12100db370d81485cdf75e47ca"]))
