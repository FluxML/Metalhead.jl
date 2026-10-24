# Ported from transformers.models.qwen2.modeling_qwen2 (transformers 5.20, Apache-2.0,
# The Hugging Face Team). Validated against test/fixtures/qwen2_* and
# test/references/Qwen--Qwen2.5-0.5B.

"""
    qwen2_for_causal_lm(; vocab_size = 151936, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `Qwen2ForCausalLM`, built from the fields of its `Qwen2Config` with the
same names and defaults, and `head_dim`, which `Qwen2Config` lacks but transformers reads.
transformers derives Qwen2 from Llama's classes, adding a bias to the query, key and value
projections, so the model is a `LlamaForCausalLM` (see
[`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm) for its inputs and
outputs). Sliding-window attention (`use_sliding_window`, `layer_types`) and rotary
embeddings other than the default, `"linear"` and `"llama3"` ones error.
"""
function qwen2_for_causal_lm(; vocab_size = 151936, hidden_size = 4096,
        intermediate_size = 22016, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = 32, hidden_act = "silu", max_position_embeddings = 32768,
        initializer_range = 0.02, rms_norm_eps = 1e-6, tie_word_embeddings = false,
        rope_theta = nothing, rope_scaling = nothing, rope_parameters = nothing,
        use_sliding_window = false, sliding_window = 4096, max_window_layers = 28,
        layer_types = nothing, attention_dropout = 0.0, head_dim = nothing)
    check_full_attention("Qwen2ForCausalLM"; num_hidden_layers, use_sliding_window,
        sliding_window, max_window_layers, layer_types)
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    rope = rope_parameters_of(
        "Qwen2ForCausalLM"; rope_theta, rope_scaling, rope_parameters,
        max_position_embeddings)
    attention(dense) = LlamaAttention(head_dim,
        dense(hidden_size => num_attention_heads * head_dim, true),
        dense(hidden_size => num_key_value_heads * head_dim, true),
        dense(hidden_size => num_key_value_heads * head_dim, true),
        dense(num_attention_heads * head_dim => hidden_size, false),
        dropout_or_identity(attention_dropout))
    return llama_like(attention; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, hidden_act, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters = rope, head_dim)
end

register!(ArchSpec(TRANSFORMERS, "Qwen2ForCausalLM", qwen2_for_causal_lm;
    family = "qwen2",
    references = ["Qwen/Qwen2.5-0.5B" => "060db6499f32faf8b98477b0a26969ef7d8b9987"]))
