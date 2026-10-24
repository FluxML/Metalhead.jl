# Ported from transformers.models.granite.modeling_granite (transformers 5.20, Apache-2.0,
# The Hugging Face Team). Validated against test/fixtures/granite_* and
# test/references/ibm-granite--granite-3.3-2b-base.

# `LlamaAttention` scaled by `attention_multiplier`, with the rotary embedding or, where the
# model has none (`GraniteMoeHybridAttention` without RoPE), no positional embedding.
@concrete struct GraniteAttention
    head_dim
    scaling
    q_proj
    k_proj
    v_proj
    o_proj
    attention_dropout
end
Flux.@layer GraniteAttention

function (m::GraniteAttention)(x, position_embeddings, kv_lengths = nothing)
    _, n, b = size(x)
    q = reshape(m.q_proj(x), m.head_dim, :, n, b)
    k = reshape(m.k_proj(x), m.head_dim, :, n, b)
    v = reshape(m.v_proj(x), m.head_dim, :, n, b)
    if position_embeddings !== nothing
        cos, sin = position_embeddings
        q = NNlib.apply_rotary_embedding(q, cos, sin)
        k = NNlib.apply_rotary_embedding(k, cos, sin)
    end
    y = NNlib.scaled_dot_product_attention(q, k, v;
        fdrop = attention_fdrop(m.attention_dropout, q),
        scale = m.scaling, is_causal = true, kv_lengths)
    return m.o_proj(reshape(y, :, n, b))
end

# `LlamaDecoderLayer` whose attention and MLP outputs are multiplied by `residual_multiplier`.
@concrete struct GraniteDecoderLayer
    self_attn
    mlp
    input_layernorm
    post_attention_layernorm
    residual_multiplier
end
Flux.@layer GraniteDecoderLayer

function (m::GraniteDecoderLayer)(x, position_embeddings, kv_lengths = nothing)
    r = eltype(x)(m.residual_multiplier)
    x = x .+ m.self_attn(m.input_layernorm(x), position_embeddings, kv_lengths) .* r
    return x .+ m.mlp(m.post_attention_layernorm(x)) .* r
end

# `LlamaModel` whose token embeddings are multiplied by `embedding_multiplier`, and whose
# `rotary_emb` is `nothing` where the attention has no positional embedding.
@concrete struct GraniteModel <: DecoderModel
    embed_tokens
    layers
    norm
    rotary_emb
    embedding_multiplier
end
Flux.@layer GraniteModel

function embed(m::GraniteModel, input_ids, attention_mask)
    x = m.embed_tokens(input_ids)
    x = x .* eltype(x)(m.embedding_multiplier)
    position_embeddings = rotary_tables(m.rotary_emb, x, size(input_ids, 1))
    return x, position_embeddings, kv_lengths_of(attention_mask, input_ids)
end

# The causal language models of Granite, whose logits are divided by `logits_scaling`.
@concrete struct GraniteForCausalLM <: CausalLM
    model
    lm_head
    logits_scaling
end
Flux.@layer GraniteForCausalLM

function forward_head(m::GraniteForCausalLM, x; pre_logits::Bool = false)
    pre_logits && return x
    logits = m.lm_head(x)
    return logits ./ eltype(logits)(m.logits_scaling)
end

"""
    granite_for_causal_lm(; vocab_size = 32000, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `GraniteForCausalLM` (IBM Granite 3 and 4.1), built from the fields of
its `GraniteConfig` with the same names and defaults, `rope_theta` and `rope_scaling`, which
older `config.json`s have in place of `rope_parameters`, and `head_dim`, which the attention
reads where a `config.json` has it. It takes and returns the arrays of
[`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm).

Granite is Llama with four multipliers: the token embeddings are multiplied by
`embedding_multiplier`, the attention scores by `attention_multiplier` (in place of
`1/√head_dim`), the outputs of the attention and of the MLP by `residual_multiplier`, and the
logits are divided by `logits_scaling`.
"""
function granite_for_causal_lm(; vocab_size = 32000, hidden_size = 4096,
        intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = nothing, head_dim = nothing, hidden_act = "silu",
        max_position_embeddings = 2048, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = false, rope_parameters = nothing, rope_theta = nothing,
        rope_scaling = nothing, attention_bias = false, attention_dropout = 0.0,
        mlp_bias = false, embedding_multiplier = 1.0, logits_scaling = 1.0,
        residual_multiplier = 1.0, attention_multiplier = 1.0)
    rope = rope_parameters_of("GraniteForCausalLM"; rope_theta, rope_scaling,
        rope_parameters, max_position_embeddings)
    act = act_fn(hidden_act)
    function decoder_layer(attention, dense, norm)
        mlp = LlamaMLP(dense(hidden_size => intermediate_size, mlp_bias),
            dense(hidden_size => intermediate_size, mlp_bias),
            dense(intermediate_size => hidden_size, mlp_bias), act)
        return GraniteDecoderLayer(attention, mlp, norm(), norm(),
            Float64(residual_multiplier))
    end
    return granite_like(decoder_layer; vocab_size, hidden_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, initializer_range,
        rms_norm_eps, tie_word_embeddings, rope_parameters = rope, attention_bias,
        attention_dropout, embedding_multiplier, logits_scaling, attention_multiplier)
end

# A `GraniteForCausalLM` whose decoder layers are `decoder_layer(attention, dense, norm)`,
# for their `GraniteAttention`, a function `dense(in => out, bias)` that builds a Linear
# layer initialised like transformers' and one `norm()` that builds an RMSNorm. With
# `rope_parameters = nothing`, the model has no rotary embedding.
function granite_like(decoder_layer; vocab_size, hidden_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, initializer_range, rms_norm_eps,
        tie_word_embeddings, rope_parameters, attention_bias, attention_dropout,
        embedding_multiplier, logits_scaling, attention_multiplier)
    # the attention takes `hidden_size ÷ num_attention_heads` features per head, and no
    # config checks that this divides
    head_dim = something(head_dim, hidden_size ÷ num_attention_heads)
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    init = normal_init(initializer_range)
    dense(in_out, bias) = Dense(in_out; bias, init)
    norm() = RMSNorm(hidden_size; eps = rms_norm_eps)
    function layer()
        attention = GraniteAttention(head_dim, Float64(attention_multiplier),
            dense(hidden_size => num_attention_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(num_attention_heads * head_dim => hidden_size, attention_bias),
            dropout_or_identity(attention_dropout))
        return decoder_layer(attention, dense, norm)
    end
    embed_tokens = Embedding(vocab_size => hidden_size; init)
    rotary_emb = rope_parameters === nothing ? nothing :
                 RotaryEmbedding(head_dim, rope_parameters)
    model = GraniteModel(embed_tokens, [layer() for _ in 1:num_hidden_layers], norm(),
        rotary_emb, Float64(embedding_multiplier))
    lm_head = tie_word_embeddings ? TiedLMHead(embed_tokens.weight) :
              dense(hidden_size => vocab_size, false)
    return GraniteForCausalLM(model, lm_head, Float64(logits_scaling))
end

register!(ArchSpec(TRANSFORMERS, "GraniteForCausalLM", granite_for_causal_lm;
    family = "granite",
    references = [
        "ibm-granite/granite-3.3-2b-base" => "c15dc681f4b2aeb4e576cdaae5434edfc17ed14d"]))
