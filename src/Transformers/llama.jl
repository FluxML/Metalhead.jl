# Ported from transformers.models.llama.modeling_llama (transformers 5.20, Apache-2.0,
# The Hugging Face Team). Validated against test/fixtures/llama_* and
# test/references/HuggingFaceTB--SmolLM2-135M.

@concrete struct LlamaMLP
    gate_proj
    up_proj
    down_proj
    act_fn
end
Flux.@layer LlamaMLP

(m::LlamaMLP)(x) = m.down_proj(m.act_fn(m.gate_proj(x)) .* m.up_proj(x))

@concrete struct LlamaAttention
    head_dim
    q_proj
    k_proj
    v_proj
    o_proj
    attention_dropout
end
Flux.@layer LlamaAttention

function (m::LlamaAttention)(x, cos, sin)
    _, n, b = size(x)
    q = reshape(m.q_proj(x), m.head_dim, :, n, b)
    k = reshape(m.k_proj(x), m.head_dim, :, n, b)
    v = reshape(m.v_proj(x), m.head_dim, :, n, b)
    q = NNlib.apply_rotary_embedding(q, cos, sin)
    k = NNlib.apply_rotary_embedding(k, cos, sin)
    y = NNlib.scaled_dot_product_attention(q, k, v; fdrop = m.attention_dropout,
        is_causal = true)
    return m.o_proj(reshape(y, :, n, b))
end

@concrete struct LlamaDecoderLayer
    self_attn
    mlp
    input_layernorm
    post_attention_layernorm
end
Flux.@layer LlamaDecoderLayer

function (m::LlamaDecoderLayer)(x, cos, sin)
    x = x .+ m.self_attn(m.input_layernorm(x), cos, sin)
    return x .+ m.mlp(m.post_attention_layernorm(x))
end

@concrete struct LlamaModel
    embed_tokens
    layers
    norm
    rotary_emb
end
Flux.@layer LlamaModel

function (m::LlamaModel)(input_ids::AbstractMatrix{<:Integer})
    x = m.embed_tokens(input_ids)
    cos, sin = rotary_tables(m.rotary_emb, x, size(input_ids, 1))
    for layer in m.layers
        x = layer(x, cos, sin)
    end
    return m.norm(x)
end

(m::LlamaModel)(input_ids::AbstractVector{<:Integer}) = dropdims(m(reshape(input_ids, :, 1)); dims = 3)

@concrete struct LlamaForCausalLM
    model
    lm_head
end
Flux.@layer LlamaForCausalLM

forward_features(m::LlamaForCausalLM, input_ids) = m.model(input_ids)

forward_head(m::LlamaForCausalLM, x; pre_logits::Bool = false) = pre_logits ? x : m.lm_head(x)

(m::LlamaForCausalLM)(input_ids) = forward_head(m, forward_features(m, input_ids))

"""
    llama_for_causal_lm(; vocab_size = 32000, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `LlamaForCausalLM`, built from the fields of its `LlamaConfig` (the
`config.json` of a checkpoint), with the same names and defaults. `create_model` ignores the
other keys of `config.json`, such as the token ids and generation settings, as transformers does.
Rotary embeddings other than the default (`rope_scaling` or a `rope_parameters` with another
`rope_type`, as in Llama 3) error.

The model maps token indices of size `(seq_len, batch_size)`, or a vector for a single
sequence, to logits of size `(vocab_size, seq_len, batch_size)`, with causal attention and no
padding. Token indices start at 1: add 1 to the ids of a Hugging Face tokenizer.
[`forward_features`](@ref) returns the hidden states after the final norm.
"""
function llama_for_causal_lm(; vocab_size = 32000, hidden_size = 4096,
        intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = nothing, hidden_act = "silu", max_position_embeddings = 2048,
        initializer_range = 0.02, rms_norm_eps = 1e-6, tie_word_embeddings = false,
        rope_theta = nothing, rope_scaling = nothing, rope_parameters = nothing,
        attention_bias = false, attention_dropout = 0.0, mlp_bias = false,
        head_dim = nothing)
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size` ($hidden_size) must be divisible by `num_attention_heads` ($num_attention_heads)")
    num_key_value_heads = something(num_key_value_heads, num_attention_heads)
    num_attention_heads % num_key_value_heads == 0 ||
        error("`num_attention_heads` ($num_attention_heads) must be divisible by `num_key_value_heads` ($num_key_value_heads)")
    head_dim = something(head_dim, hidden_size ÷ num_attention_heads)
    theta = rope_theta_of("LlamaForCausalLM"; rope_theta, rope_scaling, rope_parameters)
    init = normal_init(initializer_range)
    dense(in_out, bias) = Dense(in_out; bias, init)
    function decoder_layer()
        attn = LlamaAttention(head_dim,
            dense(hidden_size => num_attention_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(num_attention_heads * head_dim => hidden_size, attention_bias),
            dropout_or_identity(attention_dropout))
        mlp = LlamaMLP(dense(hidden_size => intermediate_size, mlp_bias),
            dense(hidden_size => intermediate_size, mlp_bias),
            dense(intermediate_size => hidden_size, mlp_bias), act_fn(hidden_act))
        return LlamaDecoderLayer(attn, mlp, RMSNorm(hidden_size; eps = rms_norm_eps),
            RMSNorm(hidden_size; eps = rms_norm_eps))
    end
    embed_tokens = Embedding(vocab_size => hidden_size; init)
    model = LlamaModel(embed_tokens, [decoder_layer() for _ in 1:num_hidden_layers],
        RMSNorm(hidden_size; eps = rms_norm_eps), RotaryEmbedding(head_dim, theta))
    lm_head = tie_word_embeddings ? TiedLMHead(embed_tokens.weight) :
              dense(hidden_size => vocab_size, false)
    return LlamaForCausalLM(model, lm_head)
end

register!(ArchSpec(TRANSFORMERS, "LlamaForCausalLM", llama_for_causal_lm;
    family = "llama",
    references = ["HuggingFaceTB/SmolLM2-135M" => "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"]))
