# Ported from transformers.models.llama.modeling_llama (transformers 5.20, Apache-2.0,
# The Hugging Face Team). Validated against test/fixtures/llama_* and
# test/references/HuggingFaceTB--SmolLM2-135M and unsloth--Llama-3.2-1B.

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

function (m::LlamaAttention)(x, position_embeddings, kv_lengths = nothing)
    _, n, b = size(x)
    cos, sin = position_embeddings
    q = reshape(m.q_proj(x), m.head_dim, :, n, b)
    k = reshape(m.k_proj(x), m.head_dim, :, n, b)
    v = reshape(m.v_proj(x), m.head_dim, :, n, b)
    q = NNlib.apply_rotary_embedding(q, cos, sin)
    k = NNlib.apply_rotary_embedding(k, cos, sin)
    y = NNlib.scaled_dot_product_attention(q, k, v;
        fdrop = attention_fdrop(m.attention_dropout, q),
        is_causal = true, kv_lengths)
    return m.o_proj(reshape(y, :, n, b))
end

@concrete struct LlamaDecoderLayer
    self_attn
    mlp
    input_layernorm
    post_attention_layernorm
end
Flux.@layer LlamaDecoderLayer

function (m::LlamaDecoderLayer)(x, position_embeddings, kv_lengths = nothing)
    x = x .+ m.self_attn(m.input_layernorm(x), position_embeddings, kv_lengths)
    return x .+ m.mlp(m.post_attention_layernorm(x))
end

# A decoder of token indices, with the fields `layers` and `norm`, whose `embed` gives the
# inputs of its first layer.
abstract type DecoderModel end

@concrete struct LlamaModel <: DecoderModel
    embed_tokens
    layers
    norm
    rotary_emb
end
Flux.@layer LlamaModel

# The token embeddings, the rotary tables of the layers (transformers'
# `position_embeddings`) and the lengths of the sequences of a batch.
function embed(m::LlamaModel, input_ids, attention_mask)
    x = m.embed_tokens(input_ids)
    position_embeddings = rotary_tables(m.rotary_emb, x, size(input_ids, 1))
    return x, position_embeddings, kv_lengths_of(attention_mask, input_ids)
end

function (m::DecoderModel)(input_ids::AbstractMatrix{<:Integer}; attention_mask = nothing)
    x, position_embeddings, kv_lengths = embed(m, input_ids, attention_mask)
    for layer in m.layers
        x = layer(x, position_embeddings, kv_lengths)
    end
    return m.norm(x)
end

function (m::DecoderModel)(input_ids::AbstractVector{<:Integer}; attention_mask = nothing)
    mask = attention_mask === nothing ? nothing : reshape(attention_mask, :, 1)
    return dropdims(m(reshape(input_ids, :, 1); attention_mask = mask); dims = 3)
end

# A causal language model: a decoder `model` (a `DecoderModel`) and the output projection
# `lm_head` of its `forward_head`.
abstract type CausalLM end

function forward_features(m::CausalLM, input_ids::AbstractArray{<:Integer};
        attention_mask = nothing)
    return m.model(input_ids; attention_mask)
end

function (m::CausalLM)(input_ids; kwargs...)
    return forward_head(m, forward_features(m, input_ids; kwargs...))
end

# The output of a processor, a NamedTuple, as the token indices and the keyword arguments
# `input_keys(model)` of a model, `nothing` for those it lacks.
function token_inputs(m, inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids,); hint = input_hint(m))
    return inputs.input_ids, (; (k => get(inputs, k, nothing) for k in input_keys(m))...)
end
input_keys(::CausalLM) = (:attention_mask,)
input_hint(::CausalLM) = "pass `text` to the processor"

function forward_features(m::CausalLM, inputs::NamedTuple)
    input_ids, kwargs = token_inputs(m, inputs)
    return forward_features(m, input_ids; kwargs...)
end

function forward_intermediates(m::CausalLM, inputs::NamedTuple; kwargs...)
    input_ids, model_kwargs = token_inputs(m, inputs)
    return forward_intermediates(m, input_ids; model_kwargs..., kwargs...)
end

function forward_intermediates(m::CausalLM, input_ids::AbstractVector{<:Integer};
        intermediates_only::Bool = false, kwargs...)
    batched = forward_intermediates(m, reshape(input_ids, :, 1); intermediates_only,
        (k => one_sequence(k, v) for (k, v) in pairs(kwargs))...)
    unbatch(x) = dropdims(x; dims = 3)
    intermediates_only && return map(unbatch, batched)
    features, intermediates = batched
    return unbatch(features), map(unbatch, intermediates)
end

# A keyword argument of the model for one sequence as that of a batch of one
one_sequence(k, v::AbstractVector) = k === :attention_mask ? reshape(v, :, 1) : v
one_sequence(k, v::AbstractMatrix) = k === :position_ids ? reshape(v, size(v)..., 1) : v
one_sequence(k, v) = v

# The decoder of a language model, and the name of its module in PyTorch
decoder(m::CausalLM) = m.model, "model"

function feature_info(m::CausalLM)
    model, name = decoder(m)
    num_chs = size(model.embed_tokens.weight, 1)
    return [(; num_chs, reduction = 1, module_name = "$name.layers.$(i - 1)")
            for i in eachindex(model.layers)]
end

@concrete struct LlamaForCausalLM <: CausalLM
    model
    lm_head
end
Flux.@layer LlamaForCausalLM

forward_head(m::LlamaForCausalLM, x; pre_logits::Bool = false) = pre_logits ? x : m.lm_head(x)

"""
    forward_intermediates(model::Union{LlamaForCausalLM, Gemma3ForCausalLM, GraniteForCausalLM, Qwen3_5ForCausalLM},
                          input_ids; indices = nothing, norm = false, stop_early = false,
                          intermediates_only = false, attention_mask = nothing)

The outputs of the decoder layers of a language model (Llama, Qwen2, Qwen3, Qwen3.5,
Gemma 3, Granite) for the token indices `input_ids` of size `(seq_len, batch_size)`, each of
size `(hidden_size, seq_len, batch_size)`, or of a vector for a single sequence, each of
size `(hidden_size, seq_len)`: the `i`-th intermediate
is transformers' `output_hidden_states[i]`, the output of the `i`-th layer, except that the
last one is not normed unless `norm = true`. The features are the hidden states after the final
norm, as [`forward_features`](@ref) returns them. `indices`, `norm`, `stop_early` and
`intermediates_only` are those of the vision models, and `attention_mask` that of the model,
as is the `position_ids` keyword of Qwen3.5. The output of a processor can replace
`input_ids` and `attention_mask`, as for the model. [`feature_info`](@ref) gives the
`hidden_size` and the module of each layer.
"""
function forward_intermediates(m::CausalLM, input_ids::AbstractMatrix{<:Integer};
        indices = nothing, norm::Bool = false, stop_early::Bool = false,
        intermediates_only::Bool = false, attention_mask = nothing)
    model = m.model
    take, max_index = feature_take_indices(length(model.layers), indices)
    x, position_embeddings, kv_lengths = embed(model, input_ids, attention_mask)
    layers = [h -> layer(h, position_embeddings, kv_lengths) for layer in model.layers]
    outs = stage_outputs(layers, x, stop_early ? max_index : length(layers))
    intermediates = [norm ? model.norm(outs[i]) : outs[i] for i in take]
    intermediates_only && return intermediates
    return model.norm(last(outs)), intermediates
end

"""
    llama_for_causal_lm(; vocab_size = 32000, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `LlamaForCausalLM`, built from the fields of its `LlamaConfig` (the
`config.json` of a checkpoint), with the same names and defaults. `create_model` ignores the
other keys of `config.json`, such as the token ids and generation settings, as transformers does.
The rotary embedding is the default one or, with `rope_scaling` or `rope_parameters`, one
with `rope_type = "linear"` or `"llama3"` (Llama 3.1 and later); other types error.

The model maps token indices of size `(seq_len, batch_size)`, or a vector for a single
sequence, to logits of size `(vocab_size, seq_len, batch_size)`, with causal attention. Token
indices start at 1: add 1 to the ids of a Hugging Face tokenizer. A batch of sequences of
different lengths, padded on the right, takes the tokenizer's `attention_mask` of size
`(seq_len, batch_size)` as a keyword argument, which keeps the padding out of the keys.
The output of its processor, a [`TextProcessor`](@ref Metalhead.TextProcessor), can replace
both: `model(tok(texts))`. [`forward_features`](@ref) returns the hidden states after the
final norm, and [`forward_intermediates`](@ref) those of each layer.
"""
function llama_for_causal_lm(; vocab_size = 32000, hidden_size = 4096,
        intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = nothing, hidden_act = "silu", max_position_embeddings = 2048,
        initializer_range = 0.02, rms_norm_eps = 1e-6, tie_word_embeddings = false,
        rope_theta = nothing, rope_scaling = nothing, rope_parameters = nothing,
        attention_bias = false, attention_dropout = 0.0, mlp_bias = false,
        head_dim = nothing)
    # `LlamaConfig` checks this even when `head_dim` is given
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size` ($hidden_size) must be divisible by `num_attention_heads` ($num_attention_heads)")
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    rope = rope_parameters_of("LlamaForCausalLM"; rope_theta, rope_scaling, rope_parameters,
        max_position_embeddings)
    attention(dense) = LlamaAttention(head_dim,
        dense(hidden_size => num_attention_heads * head_dim, attention_bias),
        dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
        dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
        dense(num_attention_heads * head_dim => hidden_size, attention_bias),
        dropout_or_identity(attention_dropout))
    return llama_like(attention; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, hidden_act, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters = rope, head_dim, mlp_bias)
end

# A `LlamaForCausalLM` whose layers have the self-attention `attention(dense)`, where
# `dense(in => out, bias)` is a Linear layer initialised like transformers', and the MLP
# `mlp(dense, i)` of the layer `i` (from 0), by default a `LlamaMLP`. The models that
# transformers derives from Llama's classes (Qwen2, Qwen3, Qwen3 MoE) are built with it.
function llama_like(attention; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, hidden_act, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters, head_dim, mlp_bias = false, mlp = nothing)
    init = normal_init(initializer_range)
    dense(in_out, bias) = init_layer(Dense, in_out; bias, init)
    norm() = RMSNorm(hidden_size; eps = rms_norm_eps)
    function llama_mlp()
        return LlamaMLP(dense(hidden_size => intermediate_size, mlp_bias),
            dense(hidden_size => intermediate_size, mlp_bias),
            dense(intermediate_size => hidden_size, mlp_bias), act_fn(hidden_act))
    end
    function decoder_layer(i)
        ffn = mlp === nothing ? llama_mlp() : mlp(dense, i)
        return LlamaDecoderLayer(attention(dense), ffn, norm(), norm())
    end
    embed_tokens = init_layer(Embedding, vocab_size => hidden_size; init)
    model = LlamaModel(
        embed_tokens, [decoder_layer(i) for i in 0:(num_hidden_layers - 1)], norm(),
        RotaryEmbedding(head_dim, rope_parameters))
    lm_head = tie_word_embeddings ? TiedLMHead(embed_tokens.weight) :
              dense(hidden_size => vocab_size, false)
    return LlamaForCausalLM(model, lm_head)
end

register!(ArchSpec(TRANSFORMERS, "LlamaForCausalLM", llama_for_causal_lm;
    family = "llama",
    references = [
        "HuggingFaceTB/SmolLM2-135M" => "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "unsloth/Llama-3.2-1B" => "9535bd9b1d1dea6acafbdc4813b728796aeb28da"]))
