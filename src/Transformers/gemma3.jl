# Ported from transformers.models.gemma3.modeling_gemma3 (transformers 5.20, Apache-2.0,
# The Hugging Face Team): the text model `Gemma3ForCausalLM`, not the multimodal
# `Gemma3ForConditionalGeneration`. Validated against test/fixtures/gemma3_* and
# test/references/unsloth--gemma-3-270m.

"""
    Gemma3TextScaledWordEmbedding(weight, embed_scale)

An `nn.Embedding` whose outputs are multiplied by `embed_scale`, `√hidden_size`, which
transformers stores in a float32 buffer and converts to the type of the weight.
"""
@concrete struct Gemma3TextScaledWordEmbedding
    weight
    embed_scale
end
Flux.@layer Gemma3TextScaledWordEmbedding

function (m::Gemma3TextScaledWordEmbedding)(input_ids)
    x = NNlib.gather(m.weight, input_ids)
    return x .* eltype(x)(m.embed_scale)
end

"""
    Gemma3RMSNorm(dim; eps)

The RMSNorm of Gemma, which scales the normalised features by `1 + weight`, its `weight`
starting at zero. As [`RMSNorm`](@ref Metalhead.Transformers.RMSNorm), it normalises in at
least `Float32` and keeps its `weight` in `Float32` under `f16`/`bf16`.
"""
@concrete struct Gemma3RMSNorm
    weight
    eps
end
Flux.@layer Gemma3RMSNorm
Flux.f16(m::Gemma3RMSNorm, cx) = f32(m, cx)
Flux.bf16(m::Gemma3RMSNorm, cx) = f32(m, cx)

Gemma3RMSNorm(dim::Integer; eps) = Gemma3RMSNorm(zeros(Float32, dim), Float64(eps))

(m::Gemma3RMSNorm)(x) = NNlib.rmsnorm(1 .+ m.weight, x; eps = m.eps)

# `Qwen3Attention` scaled by `query_pre_attn_scalar^-0.5`, with the rotary tables of its
# `layer_type` and, if it is `:sliding_attention`, a window of `sliding_window` keys.
@concrete struct Gemma3Attention
    layer_type
    head_dim
    scaling
    sliding_window
    q_proj
    k_proj
    v_proj
    o_proj
    q_norm
    k_norm
    attention_dropout
end
Flux.@layer Gemma3Attention

function (m::Gemma3Attention)(x, position_embeddings, kv_lengths = nothing)
    _, n, b = size(x)
    cos, sin = position_embeddings[m.layer_type]
    q = m.q_norm(reshape(m.q_proj(x), m.head_dim, :, n, b))
    k = m.k_norm(reshape(m.k_proj(x), m.head_dim, :, n, b))
    v = reshape(m.v_proj(x), m.head_dim, :, n, b)
    q = NNlib.apply_rotary_embedding(q, cos, sin)
    k = NNlib.apply_rotary_embedding(k, cos, sin)
    y = NNlib.scaled_dot_product_attention(q, k, v;
        fdrop = attention_fdrop(m.attention_dropout, q),
        scale = m.scaling, is_causal = true, kv_lengths,
        window = attention_window(n, m.sliding_window),
        q_lengths = sliding_window_q_lengths(kv_lengths, n, m.sliding_window))
    return m.o_proj(reshape(y, :, n, b))
end

# Gemma's decoder layer normalises the outputs of the attention and of the MLP too.
@concrete struct Gemma3DecoderLayer
    self_attn
    mlp
    input_layernorm
    post_attention_layernorm
    pre_feedforward_layernorm
    post_feedforward_layernorm
end
Flux.@layer Gemma3DecoderLayer

function (m::Gemma3DecoderLayer)(x, position_embeddings, kv_lengths = nothing)
    h = m.self_attn(m.input_layernorm(x), position_embeddings, kv_lengths)
    x = x .+ m.post_attention_layernorm(h)
    h = m.mlp(m.pre_feedforward_layernorm(x))
    return x .+ m.post_feedforward_layernorm(h)
end

# Gemma3RotaryEmbedding: the rotary embeddings of each layer type, in a NamedTuple.
function rotary_tables(m::NamedTuple, x, n)
    return map(r -> rotary_tables(r, x, n), m)
end

@concrete struct Gemma3ForCausalLM <: CausalLM
    model
    lm_head
    final_logit_softcapping
end
Flux.@layer Gemma3ForCausalLM

function forward_head(m::Gemma3ForCausalLM, x; pre_logits::Bool = false)
    pre_logits && return x
    logits = m.lm_head(x)
    m.final_logit_softcapping === nothing && return logits
    cap = eltype(logits)(m.final_logit_softcapping)
    return tanh.(logits ./ cap) .* cap
end

"""
    gemma3_for_causal_lm(; vocab_size = 262208, hidden_size = 2304, num_hidden_layers = 26, kwargs...)

Port of transformers' `Gemma3ForCausalLM`, the Gemma 3 text model, built from the fields of
its `Gemma3TextConfig` with the same names and defaults, and `rope_theta`,
`rope_local_base_freq`, `rope_scaling` and `sliding_window_pattern`, which older
`config.json`s have in place of `rope_parameters` and `layer_types`. It takes and returns
the arrays of [`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm).

Gemma 3 differs from Llama in its decoder layers: the token embeddings are multiplied by
`√hidden_size`, the RMSNorms scale by `1 + weight`, the outputs of the attention and of the
MLP are normalised too, the queries and keys are normalised per head and scaled by
`query_pre_attn_scalar^-0.5`, the MLP gate is `gelu_pytorch_tanh`, and the logits may be
softcapped (`final_logit_softcapping`). The layers attend either to all the previous tokens
(`"full_attention"`) or to the last `sliding_window` of them (`"sliding_attention"`), as
`layer_types` says, with rotary embeddings of their own: `rope_parameters` has one for each
layer type. Bidirectional attention (`use_bidirectional_attention`) errors.
`attn_logit_softcapping` is ignored, as transformers ignores it.
"""
function gemma3_for_causal_lm(; vocab_size = 262_208, hidden_size = 2304,
        intermediate_size = 9216, num_hidden_layers = 26, num_attention_heads = 8,
        num_key_value_heads = 4, head_dim = 256, hidden_activation = "gelu_pytorch_tanh",
        max_position_embeddings = 131_072, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = true, rope_parameters = nothing, rope_theta = nothing,
        rope_local_base_freq = nothing, rope_scaling = nothing, attention_bias = false,
        attention_dropout = 0.0, query_pre_attn_scalar = 256, sliding_window = 4096,
        layer_types = nothing, sliding_window_pattern = 6,
        final_logit_softcapping = nothing, use_bidirectional_attention = false)
    name = "Gemma3ForCausalLM"
    something(use_bidirectional_attention, false) &&
        error("`use_bidirectional_attention = true` is not supported by the Metalhead port of $name")
    # `Gemma3TextConfig` checks this even with `head_dim`
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size` ($hidden_size) must be divisible by `num_attention_heads` ($num_attention_heads)")
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    types = gemma3_layer_types(; num_hidden_layers, layer_types, sliding_window_pattern,
        sliding_window)
    rope = gemma3_rope_parameters(; rope_parameters, rope_theta, rope_local_base_freq,
        rope_scaling, max_position_embeddings)
    act = act_fn(hidden_activation; key = "hidden_activation")
    init = normal_init(initializer_range)
    dense(in_out, bias) = Dense(in_out; bias, init)
    norm(dim) = Gemma3RMSNorm(dim; eps = rms_norm_eps)
    function decoder_layer(layer_type)
        window = layer_type == :sliding_attention ? sliding_window : nothing
        attention = Gemma3Attention(layer_type, head_dim,
            Float64(query_pre_attn_scalar)^-0.5, window,
            dense(hidden_size => num_attention_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(num_attention_heads * head_dim => hidden_size, attention_bias),
            norm(head_dim), norm(head_dim), dropout_or_identity(attention_dropout))
        mlp = LlamaMLP(dense(hidden_size => intermediate_size, false),
            dense(hidden_size => intermediate_size, false),
            dense(intermediate_size => hidden_size, false), act)
        return Gemma3DecoderLayer(attention, mlp, norm(hidden_size), norm(hidden_size),
            norm(hidden_size), norm(hidden_size))
    end
    embed_tokens = Gemma3TextScaledWordEmbedding(init(hidden_size, vocab_size),
        Float32(sqrt(hidden_size)))
    model = LlamaModel(embed_tokens, [decoder_layer(t) for t in types], norm(hidden_size),
        map(p -> RotaryEmbedding(head_dim, p), rope))
    lm_head = tie_word_embeddings ? TiedLMHead(embed_tokens.weight) :
              dense(hidden_size => vocab_size, false)
    softcap = final_logit_softcapping === nothing ? nothing :
              Float64(final_logit_softcapping)
    return Gemma3ForCausalLM(model, lm_head, softcap)
end

# `layer_types` as Symbols, computed from `sliding_window_pattern` as `Gemma3TextConfig`
# does when the config has none: every `sliding_window_pattern`-th layer has full attention.
function gemma3_layer_types(; num_hidden_layers, layer_types, sliding_window_pattern,
        sliding_window)
    types = something(layer_types,
        [(i + 1) % sliding_window_pattern == 0 ? "full_attention" : "sliding_attention"
         for i in 0:(num_hidden_layers - 1)])
    length(types) == num_hidden_layers ||
        error("`layer_types` has $(length(types)) entries, `num_hidden_layers` is $num_hidden_layers")
    supported = ("full_attention", "sliding_attention")
    for t in types
        t in supported ||
            error("`layer_types` entry $(repr(t)) is not supported; supported: $supported")
    end
    "sliding_attention" in types && sliding_window === nothing &&
        error("`sliding_window` is required by the \"sliding_attention\" layers")
    return Symbol.(types)
end

# The rotary embeddings of the two layer types, as in
# `Gemma3TextConfig.convert_rope_params_to_dict`: a `rope_parameters` nested by layer type,
# where `rope_scaling` updates that of the full attention, and `rope_theta` and
# `rope_local_base_freq` give the default `rope_theta` of the full and sliding-window
# attention.
function gemma3_rope_parameters(; rope_parameters, rope_theta, rope_local_base_freq,
        rope_scaling, max_position_embeddings)
    params = string_keys(something(rope_parameters, (;)))
    of_type(t) = string_keys(something(get(params, t, nothing), (; rope_type = "default")))
    full, sliding = of_type("full_attention"), of_type("sliding_attention")
    rope_scaling === nothing || merge!(full, string_keys(rope_scaling))
    name = "Gemma3ForCausalLM"
    return (;
        full_attention = rope_type_parameters(name, full;
            rope_theta = something(rope_theta, 1_000_000.0), max_position_embeddings),
        sliding_attention = rope_type_parameters(name, sliding;
            rope_theta = something(rope_local_base_freq, 10_000.0),
            max_position_embeddings))
end

string_keys(d) = Dict{String, Any}(string(k) => v for (k, v) in pairs(d))

register!(ArchSpec(TRANSFORMERS, "Gemma3ForCausalLM", gemma3_for_causal_lm;
    family = "gemma3",
    references = ["unsloth/gemma-3-270m" => "edcf1fe394ee1168e7c5538afd104cfb1f7947e6"]))
