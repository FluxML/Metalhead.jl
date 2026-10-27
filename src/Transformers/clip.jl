# Ported from transformers.models.clip.modeling_clip (transformers 5.20, Apache-2.0, The
# Hugging Face Team): the text encoder of CLIP, `CLIPTextModelWithProjection`, which SAM 3
# uses. Validated against test/fixtures/sam3_*.

# The activations of the configs of SAM 3 and CLIP, by name (transformers' `ACT2FN`).
const SAM3_ACT2FN = Dict("gelu" => gelu_erf, "relu" => relu, "quick_gelu" => quick_gelu,
    "gelu_pytorch_tanh" => gelu_tanh, "silu" => swish)

@concrete struct CLIPTextEmbeddings
    token_embedding
    position_embedding
end
Flux.@layer CLIPTextEmbeddings

function (m::CLIPTextEmbeddings)(input_ids)
    n = size(input_ids, 1)
    max_positions = size(m.position_embedding.weight, 2)
    n <= max_positions ||
        error("the sequences have $n tokens, more than `max_position_embeddings = $max_positions`")
    return m.token_embedding(input_ids) .+ m.position_embedding.weight[:, 1:n]
end

@concrete struct CLIPAttention
    num_heads
    k_proj
    v_proj
    q_proj
    out_proj
    dropout
end
Flux.@layer CLIPAttention

# Causal self-attention over the first `kv_lengths` tokens of each sequence (all without).
function (m::CLIPAttention)(x, kv_lengths)
    _, n, b = size(x)
    heads(p) = reshape(p(x), :, m.num_heads, n, b)
    q, k, v = heads(m.q_proj), heads(m.k_proj), heads(m.v_proj)
    y = NNlib.scaled_dot_product_attention(q, k, v; is_causal = true, kv_lengths,
        fdrop = attention_fdrop(m.dropout, q))
    return m.out_proj(reshape(y, :, n, b))
end

@concrete struct CLIPMLP
    activation_fn
    fc1
    fc2
end
Flux.@layer CLIPMLP

(m::CLIPMLP)(x) = m.fc2(m.activation_fn(m.fc1(x)))

@concrete struct CLIPEncoderLayer
    self_attn
    layer_norm1
    mlp
    layer_norm2
end
Flux.@layer CLIPEncoderLayer

function (m::CLIPEncoderLayer)(x, kv_lengths)
    x = x .+ m.self_attn(m.layer_norm1(x), kv_lengths)
    return x .+ m.mlp(m.layer_norm2(x))
end

@concrete struct CLIPEncoder
    layers
end
Flux.@layer CLIPEncoder

function (m::CLIPEncoder)(x, kv_lengths)
    for layer in m.layers
        x = layer(x, kv_lengths)
    end
    return x
end

@concrete struct CLIPTextModel
    embeddings
    encoder
    final_layer_norm
    eos_token_id
end
Flux.@layer CLIPTextModel

# The position of the end-of-text token of each sequence, of whose features CLIP pools: the
# first token `eos_token_id`, or the largest index for configs older than transformers'
# PR #24773 (`eos_token_id = 2`), as transformers does.
function eos_positions(input_ids, eos_token_id)
    ids = Array(input_ids)
    eos_token_id == 2 && return [argmax(c) for c in eachcol(ids)]
    return [something(findfirst(==(eos_token_id + 1), c), 1) for c in eachcol(ids)]
end
ChainRulesCore.@non_differentiable eos_positions(::Any...)

"""
    (model::CLIPTextModel)(input_ids; attention_mask = nothing)

The `last_hidden_state` (after the final norm) and the `pooler_output` (its features at the
end-of-text token of each sequence) of the token indices `input_ids` (seq_len, batch_size),
1-based, padded on the right after the ones of `attention_mask`.
"""
function (m::CLIPTextModel)(input_ids; attention_mask = nothing)
    kv_lengths = kv_lengths_of(attention_mask, input_ids)
    x = m.final_layer_norm(m.encoder(m.embeddings(input_ids), kv_lengths))
    d, n, b = size(x)
    index = ChainRulesCore.ignore_derivatives() do
        device_indices(x, eos_positions(input_ids, m.eos_token_id) .+ n .* (0:(b - 1)))
    end
    return (; last_hidden_state = x, pooler_output = NNlib.gather(reshape(x, d, :), index))
end

@concrete struct CLIPTextModelWithProjection
    text_model
    text_projection
end
Flux.@layer CLIPTextModelWithProjection

"""
    (model::CLIPTextModelWithProjection)(input_ids; attention_mask = nothing)

The `text_embeds` (the projected `pooler_output`) and the `last_hidden_state` of the text
model (see [`clip_text_model_with_projection`](@ref
Metalhead.Transformers.clip_text_model_with_projection)).
"""
function (m::CLIPTextModelWithProjection)(input_ids; attention_mask = nothing)
    y = m.text_model(input_ids; attention_mask)
    return (; text_embeds = m.text_projection(y.pooler_output),
        last_hidden_state = y.last_hidden_state)
end

"""
    clip_text_model_with_projection(; vocab_size = 49408, hidden_size = 512,
        intermediate_size = 2048, projection_dim = 512, num_hidden_layers = 12,
        num_attention_heads = 8, max_position_embeddings = 77, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, eos_token_id = 49407,
        initializer_range = 0.02)

Port of transformers' `CLIPTextModelWithProjection`, the text encoder of CLIP, built from
the fields of a `CLIPTextConfig`: a causal transformer over the token indices
`(seq_len, batch_size)` (1-based), whose features at the end-of-text token are projected to
`projection_dim` (`text_embeds`).
"""
function clip_text_model_with_projection(; vocab_size = 49408, hidden_size = 512,
        intermediate_size = 2048, projection_dim = 512, num_hidden_layers = 12,
        num_attention_heads = 8, max_position_embeddings = 77, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, eos_token_id = 49407,
        initializer_range = 0.02)
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size = $hidden_size` must be divisible by `num_attention_heads = $num_attention_heads`")
    act = act_fn(hidden_act; table = SAM3_ACT2FN)
    init = normal_init(initializer_range)
    dense(io; bias = true) = init_layer(Dense, io; bias, init)
    norm() = init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps))
    layers = [CLIPEncoderLayer(
                  CLIPAttention(num_attention_heads, dense(hidden_size => hidden_size),
                      dense(hidden_size => hidden_size), dense(hidden_size => hidden_size),
                      dense(hidden_size => hidden_size),
                      dropout_or_identity(attention_dropout)),
                  norm(),
                  CLIPMLP(act, dense(hidden_size => intermediate_size),
                      dense(intermediate_size => hidden_size)),
                  norm())
              for _ in 1:num_hidden_layers]
    embeddings = CLIPTextEmbeddings(init_layer(Embedding, vocab_size => hidden_size; init),
        init_layer(Embedding, max_position_embeddings => hidden_size; init))
    text_model = CLIPTextModel(embeddings, CLIPEncoder(layers), norm(), eos_token_id)
    return CLIPTextModelWithProjection(text_model,
        dense(hidden_size => projection_dim; bias = false))
end
