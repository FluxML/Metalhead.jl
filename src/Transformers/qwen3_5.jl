# Ported from transformers.models.qwen3_5.modeling_qwen3_5 (transformers 5.20, Apache-2.0,
# The Hugging Face Team): the text model `Qwen3_5ForCausalLM`, a hybrid of gated attention
# and gated delta rule (linear attention) layers. Validated against
# test/fixtures/qwen3_5_text_*. The cache of generation is not ported.

"""
    Qwen3_5TextRotaryEmbedding(rotary_dim, rope_theta, mrope_section)

The interleaved multimodal rotary embedding (MRoPE) of Qwen3.5, which rotates the first
`rotary_dim` features of each head. Its frequency `j` (from 0) turns with the height
position if `j % 3 == 1` and `j < 3 * mrope_section[2]`, with the width position if
`j % 3 == 2` and `j < 3 * mrope_section[3]`, and with the temporal position otherwise; the
three are equal for text.
"""
@concrete struct Qwen3_5TextRotaryEmbedding
    rotary_dim
    rope_theta
    mrope_section
end
Flux.@layer Qwen3_5TextRotaryEmbedding

# The cosines and sines (rotary_dim ÷ 2, 1, seq_len, batch) of the positions
# `position_ids`, of size (3, seq_len, batch) or (3, seq_len, 1), in the element type and on
# the device of the hidden states `x`. A first row of four is transformers' text position,
# which only its attention mask reads.
function rotary_tables(m::Qwen3_5TextRotaryEmbedding, x, position_ids)
    return ChainRulesCore.ignore_derivatives() do
        n, b = size(x, 2), size(x, 3)
        pos = position_ids === nothing ? repeat(reshape(0:(n - 1), 1, n, 1), 3) :
              Array(position_ids)
        size(pos, 1) == 4 && (pos = pos[2:4, :, :])
        size(pos, 1) == 3 && size(pos, 2) == n && size(pos, 3) in (1, b) ||
            error("`position_ids` must have size (3, $n, $b), got $(size(pos))")
        freqs = inv_freq((; rope_type = "default", m.rope_theta), m.rotary_dim)
        section = m.mrope_section
        row(j) = j % 3 == 1 && j < 3 * section[2] ? 2 :
                 j % 3 == 2 && j < 3 * section[3] ? 3 : 1
        angles = [freqs[j + 1] * pos[row(j), t, i]
                  for j in 0:(length(freqs) - 1), _ in 1:1, t in 1:n, i in 1:size(pos, 3)]
        T = eltype(x)
        return to_device_of(x, T.(cos.(angles))), to_device_of(x, T.(sin.(angles)))
    end
end

# transformers' `apply_rotary_pos_emb` with partial rotation: the first `2 * size(cos, 1)`
# features of each head of `x` (head_dim, heads, seq_len, batch) are rotated, the feature
# `k` with `k + size(cos, 1)` (`rotate_half`), and the others are kept.
function apply_partial_rotary(x, cos, sin)
    h = size(cos, 1)
    x1, x2 = x[1:h, :, :, :], x[(h + 1):(2h), :, :, :]
    return cat(x1 .* cos .- x2 .* sin, x2 .* cos .+ x1 .* sin, x[(2h + 1):end, :, :, :];
        dims = 1)
end

# `Qwen3Attention` whose `q_proj` also gives a gate for each head, multiplying the output by
# its sigmoid, and whose rotary embedding is partial.
@concrete struct Qwen3_5Attention
    head_dim
    q_proj
    k_proj
    v_proj
    o_proj
    q_norm
    k_norm
    attention_dropout
end
Flux.@layer Qwen3_5Attention

function (m::Qwen3_5Attention)(x, position_embeddings, kv_lengths = nothing)
    _, n, b = size(x)
    query_gate = reshape(m.q_proj(x), m.head_dim, 2, :, n, b)
    q = m.q_norm(query_gate[:, 1, :, :, :])
    gate = reshape(query_gate[:, 2, :, :, :], :, n, b)
    k = m.k_norm(reshape(m.k_proj(x), m.head_dim, :, n, b))
    v = reshape(m.v_proj(x), m.head_dim, :, n, b)
    cos, sin = position_embeddings
    q = apply_partial_rotary(q, cos, sin)
    k = apply_partial_rotary(k, cos, sin)
    y = NNlib.scaled_dot_product_attention(q, k, v;
        fdrop = attention_fdrop(m.attention_dropout, q),
        is_causal = true, kv_lengths)
    return m.o_proj(reshape(y, :, n, b) .* sigmoid.(gate))
end

"""
    Qwen3_5RMSNormGated(dim; eps)

The RMSNorm of the output of the gated delta rule, multiplied by `activation(gate)` (SiLU),
computed in at least `Float32`. Under `f16`/`bf16` its `weight` stays `Float32`.
"""
@concrete struct Qwen3_5RMSNormGated
    weight
    variance_epsilon
    activation
end
Flux.@layer Qwen3_5RMSNormGated
Flux.f16(m::Qwen3_5RMSNormGated, cx) = f32(m, cx)
Flux.bf16(m::Qwen3_5RMSNormGated, cx) = f32(m, cx)

function Qwen3_5RMSNormGated(dim::Integer; eps)
    return Qwen3_5RMSNormGated(ones(Float32, dim), Float64(eps), swish)
end

function (m::Qwen3_5RMSNormGated)(x, gate)
    T = promote_type(eltype(x), Float32)
    y = NNlib.rmsnorm(m.weight, x; eps = m.variance_epsilon) .* m.activation.(T.(gate))
    return eltype(y) == eltype(x) ? y : eltype(x).(y)
end

"""
    Qwen3_5GatedDeltaNet

The linear attention of Qwen3.5: a gated delta rule (Yang et al., 2024) over the queries,
keys and values of `in_proj_qkv`, after a causal depthwise convolution `conv1d`. Its
recurrent state, `(head_v_dim, head_k_dim)` per value head, decays by
`exp(-exp(A_log) * softplus(in_proj_a(x) + dt_bias))` at each token and is written with
the rate `sigmoid(in_proj_b(x))`. The output is normed and gated by `in_proj_z(x)`.
"""
@concrete struct Qwen3_5GatedDeltaNet
    num_k_heads
    num_v_heads
    head_k_dim
    head_v_dim
    conv1d
    activation
    dt_bias
    A_log
    norm
    out_proj
    in_proj_qkv
    in_proj_z
    in_proj_b
    in_proj_a
end
Flux.@layer Qwen3_5GatedDeltaNet

function (m::Qwen3_5GatedDeltaNet)(x, kv_lengths = nothing)
    x = mask_padding(x, kv_lengths)
    _, n, b = size(x)
    key_dim = m.head_k_dim * m.num_k_heads
    mixed = permutedims(m.conv1d(permutedims(m.in_proj_qkv(x), (2, 1, 3))), (2, 1, 3))
    qkv = m.activation.(mixed)
    r = m.num_v_heads ÷ m.num_k_heads
    q = repeat_heads(reshape(qkv[1:key_dim, :, :], m.head_k_dim, :, n, b), r)
    k = repeat_heads(reshape(qkv[(key_dim + 1):(2key_dim), :, :], m.head_k_dim, :, n, b), r)
    v = reshape(qkv[(2key_dim + 1):end, :, :], m.head_v_dim, :, n, b)
    z = reshape(m.in_proj_z(x), m.head_v_dim, :, n, b)
    beta = sigmoid.(m.in_proj_b(x))
    # in at least Float32, as transformers' `.float()`
    T = promote_type(eltype(x), Float32)
    g = .-exp.(T.(m.A_log)) .* softplus.(T.(m.in_proj_a(x)) .+ T.(m.dt_bias))
    y = chunk_gated_delta_rule(q, k, v, g, beta)
    y = m.norm(reshape(y, m.head_v_dim, :), reshape(z, m.head_v_dim, :))
    return m.out_proj(reshape(y, :, n, b))
end

# transformers' `apply_mask_to_padding_states`: zero the padding of a batch padded on the
# right before the linear attention, whose state would carry it to the following tokens.
function mask_padding(x, kv_lengths)
    n = size(x, 2)
    (kv_lengths === nothing || all(==(n), kv_lengths)) && return x
    mask = ChainRulesCore.ignore_derivatives() do
        to_device_of(x, eltype(x).(reshape((1:n) .<= kv_lengths', 1, n, :)))
    end
    return x .* mask
end

# torch's `x.repeat_interleave(r, dim=2)` for the heads of `x` (head_dim, heads, n, batch).
function repeat_heads(x, r)
    r == 1 && return x
    d, h, n, b = size(x)
    return reshape(repeat(reshape(x, d, 1, h, n, b), 1, r, 1, 1, 1), d, r * h, n, b)
end

"""
    chunk_gated_delta_rule(q, k, v, g, beta; chunk_size = 64)

transformers' `torch_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel = true`, for the
queries and keys `q` and `k` (head_k_dim, heads, seq_len, batch), the values `v`
(head_v_dim, heads, seq_len, batch), the log decays `g` and the rates `beta`
(heads, seq_len, batch). For each head, with `q` and `k` normalised and `q` scaled by
`head_k_dim^-0.5`, the state `S` (head_v_dim, head_k_dim) starts at zero and at each token
becomes `S = exp(g) * S` then `S += beta * (v - S * k) * kᵀ`, and the output is `S * q`.

As in transformers, the sequence is padded to chunks of `chunk_size` tokens, computed in
parallel but for the state carried from one chunk to the next, in at least `Float32`. Each
chunk solves a unit triangular system, here by inverting its matrix by blocks, which runs
on a GPU and is differentiable. The output has the element type of `v`.
"""
function chunk_gated_delta_rule(q, k, v, g, beta; chunk_size = 64)
    Tout = eltype(v)
    T = promote_type(Tout, Float32)
    dk, h, n, b = size(q)
    dv = size(v, 1)
    C = chunk_size
    pad = mod(-n, C)
    nc = (n + pad) ÷ C
    chunked(x) = to_chunks(x, C, pad)
    q = chunked(l2norm(T.(q)) .* T(dk^-0.5))
    k = chunked(l2norm(T.(k)))
    v = chunked(T.(v))
    g = chunked(reshape(T.(g), 1, h, n, b))
    beta = chunked(reshape(T.(beta), 1, h, n, b))
    # all of size (features, chunk_size, heads * batch, chunks)
    k_beta = k .* beta
    cum = cumsum(g; dims = 2)
    # [j, i] = exp(cum[i] - cum[j]) for the keys j ≤ i, else 0
    decay = exp.(cum .- reshape(cum, C, 1, :, nc) .+ causal_log_mask(q, C))
    flat(x) = reshape(x, size(x, 1), size(x, 2), :)
    attn = batched_mul(batched_transpose(flat(k)), flat(q)) .* flat(decay)
    system = batched_mul(batched_transpose(flat(k)), flat(k_beta)) .* flat(decay)
    solve = unit_triangular_inverse(system)
    new_values = batched_mul(flat(v .* beta), solve)
    k_cumdecay = batched_mul(flat(k_beta .* exp.(cum)), solve)
    last_cum = cum[:, C:C, :, :]
    states = chunk_states(reshape(new_values, dv, C, :, nc),
        reshape(k_cumdecay, dk, C, :, nc), k .* exp.(last_cum .- cum), exp.(last_cum))
    s = reshape(states, dv, dk, :)
    v_new = new_values .- batched_mul(s, k_cumdecay)
    y = batched_mul(s, flat(q .* exp.(cum))) .+ batched_mul(v_new, attn)
    y = reshape(permutedims(reshape(y, dv, C, h, b, nc), (1, 3, 2, 5, 4)), dv, h, :, b)
    y = pad == 0 ? y : y[:, :, 1:n, :]
    return Tout == T ? y : Tout.(y)
end

# transformers' `l2norm` over the features
l2norm(x; eps = 1e-6) = x .* inv.(sqrt.(sum(abs2, x; dims = 1) .+ eltype(x)(eps)))

# `x` (features, heads, seq_len, batch) padded with `pad` zero tokens and cut in chunks of
# `C` tokens: (features, C, heads * batch, chunks).
function to_chunks(x, C, pad)
    d, h, _, b = size(x)
    x = pad == 0 ? x : NNlib.pad_zeros(x, (0, pad); dims = 3)
    x = permutedims(reshape(x, d, h, C, :, b), (1, 3, 2, 5, 4))
    return reshape(x, d, C, h * b, :)
end

# 0 where the key j comes at or before the query i, -Inf after it: (C, C), on the device
# of `x`.
function causal_log_mask(x, C)
    return ChainRulesCore.ignore_derivatives() do
        T = eltype(x)
        to_device_of(x, [j <= i ? zero(T) : T(-Inf) for j in 1:C, i in 1:C])
    end
end

# The inverse of `I + U` for the strictly upper triangular part `U` of each matrix of `a`
# (C, C, n), C a power of 2. It inverts diagonal blocks of doubling size: a block
# [A B; 0 D] whose diagonal blocks are inverted has the inverse
# [A⁻¹ -A⁻¹BD⁻¹; 0 D⁻¹], so that with `x` the block diagonal matrix of the inverses and
# `B` the off-diagonal blocks of `U`, the next `x` is `x - x B x`. Its intermediate results
# are blocks of the inverse, unlike those of the series `I - U + U² - ...`, which can be
# much larger and lose precision to cancellation.
function unit_triangular_inverse(a)
    C = size(a, 1)
    ispow2(C) || error("the chunk size must be a power of 2, got $C")
    x = identity_matrix(a, C) .- a .* block_mask(a, C, 1)
    s = 2
    while s < C
        x = x .- batched_mul(batched_mul(x, a .* block_mask(a, C, s)), x)
        s *= 2
    end
    return x
end

function identity_matrix(x, C)
    return ChainRulesCore.ignore_derivatives() do
        to_device_of(x, [eltype(x)(i == j) for i in 1:C, j in 1:C])
    end
end

# 1 at the rows of the first half and the columns of the second half of each diagonal block
# of size 2s, else 0.
function block_mask(x, C, s)
    return ChainRulesCore.ignore_derivatives() do
        mask = [(i - 1) ÷ 2s == (j - 1) ÷ 2s && iseven((i - 1) ÷ s) &&
                isodd((j - 1) ÷ s) for i in 1:C, j in 1:C]
        to_device_of(x, eltype(x).(mask))
    end
end

# The recurrent states (head_v_dim, head_k_dim, n, chunks) of the gated delta rule at the
# start of each chunk, from the values to write `u` (head_v_dim, chunk_size, n, chunks), the
# keys that read the state `w` and that write it `kd` (head_k_dim, chunk_size, n, chunks),
# and the decays `decay` (1, 1, n, chunks) of the chunks. The state starts at zero, and
# after a chunk is `decay .* S + (u - S w) kdᵀ`, as in the loop over chunks of
# transformers. Its gradient runs the loop backwards.
function chunk_states(u, w, kd, decay)
    dv, dk, n, nc = size(u, 1), size(w, 1), size(u, 3), size(u, 4)
    states = similar(u, dv, dk, n, nc)
    s = fill!(similar(u, dv, dk, n), 0)
    for c in 1:nc
        copyto!(view(states, :, :, :, c), s)
        c == nc && break
        v_new = u[:, :, :, c] .- batched_mul(s, w[:, :, :, c])
        s = s .* decay[:, :, :, c] .+ batched_mul(v_new, batched_transpose(kd[:, :, :, c]))
    end
    return states
end

function ChainRulesCore.rrule(::typeof(chunk_states), u, w, kd, decay)
    states = chunk_states(u, w, kd, decay)
    function chunk_states_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        Δ isa ChainRulesCore.AbstractZero &&
            return (ChainRulesCore.NoTangent(), ntuple(_ -> Δ, 4)...)
        nc = size(u, 4)
        du, dw, dkd, ddecay = zero(u), zero(w), zero(kd), zero(decay)
        # the gradient of the state at the start of chunk c + 1
        g = Δ[:, :, :, nc]
        for c in (nc - 1):-1:1
            s, wc = states[:, :, :, c], w[:, :, :, c]
            v_new = u[:, :, :, c] .- batched_mul(s, wc)
            dv_new = batched_mul(g, kd[:, :, :, c])
            copyto!(view(du, :, :, :, c), dv_new)
            copyto!(view(dw, :, :, :, c), .-batched_mul(batched_transpose(s), dv_new))
            copyto!(view(dkd, :, :, :, c), batched_mul(batched_transpose(g), v_new))
            copyto!(view(ddecay, :, :, :, c), sum(g .* s; dims = (1, 2)))
            g = Δ[:, :, :, c] .+ g .* decay[:, :, :, c] .-
                batched_mul(dv_new, batched_transpose(wc))
        end
        return ChainRulesCore.NoTangent(), du, dw, dkd, ddecay
    end
    return states, chunk_states_pullback
end

# A decoder layer has either a linear attention `linear_attn` or a full attention
# `self_attn`, the other being `nothing`.
@concrete struct Qwen3_5DecoderLayer
    linear_attn
    self_attn
    mlp
    input_layernorm
    post_attention_layernorm
end
Flux.@layer Qwen3_5DecoderLayer

function (m::Qwen3_5DecoderLayer)(x, position_embeddings, kv_lengths = nothing)
    h = m.input_layernorm(x)
    h = m.self_attn === nothing ? m.linear_attn(h, kv_lengths) :
        m.self_attn(h, position_embeddings, kv_lengths)
    x = x .+ h
    return x .+ m.mlp(m.post_attention_layernorm(x))
end

@concrete struct Qwen3_5TextModel
    embed_tokens
    layers
    norm
    rotary_emb
end
Flux.@layer Qwen3_5TextModel

function (m::Qwen3_5TextModel)(input_ids::AbstractMatrix{<:Integer};
        attention_mask = nothing, position_ids = nothing)
    return qwen3_5_decode(m, m.embed_tokens(input_ids), position_ids; attention_mask)
end

function (m::Qwen3_5TextModel)(input_ids::AbstractVector{<:Integer};
        attention_mask = nothing, position_ids = nothing)
    mask = attention_mask === nothing ? nothing : reshape(attention_mask, :, 1)
    pos = position_ids === nothing ? nothing :
          reshape(position_ids, size(position_ids, 1), :, 1)
    y = m(reshape(input_ids, :, 1); attention_mask = mask, position_ids = pos)
    return dropdims(y; dims = 3)
end

"""
    qwen3_5_decode(model::Qwen3_5TextModel, inputs_embeds, position_ids; attention_mask = nothing)

The hidden states (hidden_size, seq_len, batch_size) after the final norm of the Qwen3.5
text model for the input embeddings `inputs_embeds` (hidden_size, seq_len, batch_size),
transformers' `Qwen3_5TextModel.forward(inputs_embeds=...)`. `position_ids` are the
temporal, height and width positions of the tokens for the rotary embedding, an integer
array (3, seq_len, batch_size) or (3, seq_len, 1) of 0-based positions (torch's
`position_ids` (3, batch_size, seq_len) permuted), or `nothing` for `0:seq_len-1` on each
row, those of text. `attention_mask` (seq_len, batch_size) marks the tokens of a batch
padded on the right, as for the token indices.
"""
function qwen3_5_decode(m::Qwen3_5TextModel, inputs_embeds, position_ids;
        attention_mask = nothing)
    x, position_embeddings, kv_lengths = decoder_inputs(m, inputs_embeds, position_ids,
        attention_mask)
    for layer in m.layers
        x = layer(x, position_embeddings, kv_lengths)
    end
    return m.norm(x)
end

# The inputs of the decoder layers: the hidden states, the rotary tables and the lengths of
# the sequences of a batch padded on the right.
function decoder_inputs(m::Qwen3_5TextModel, inputs_embeds, position_ids, attention_mask)
    kv_lengths = kv_lengths_of(attention_mask, view(inputs_embeds, 1, :, :))
    position_embeddings = rotary_tables(m.rotary_emb, inputs_embeds, position_ids)
    return inputs_embeds, position_embeddings, kv_lengths
end

@concrete struct Qwen3_5ForCausalLM <: CausalLM
    model
    lm_head
end
Flux.@layer Qwen3_5ForCausalLM

function forward_features(m::Qwen3_5ForCausalLM, input_ids; attention_mask = nothing,
        position_ids = nothing)
    return m.model(input_ids; attention_mask, position_ids)
end

function (m::Qwen3_5ForCausalLM)(input_ids::AbstractArray{<:Integer};
        attention_mask = nothing, position_ids = nothing)
    return forward_head(m, forward_features(m, input_ids; attention_mask, position_ids))
end

function forward_head(m::Qwen3_5ForCausalLM, x; pre_logits::Bool = false)
    return pre_logits ? x : m.lm_head(x)
end

function forward_intermediates(m::Qwen3_5ForCausalLM, input_ids::AbstractMatrix{<:Integer};
        indices = nothing, norm::Bool = false, stop_early::Bool = false,
        intermediates_only::Bool = false, attention_mask = nothing, position_ids = nothing)
    return decoder_intermediates(m.model, m.model.embed_tokens(input_ids), position_ids,
        attention_mask; indices, norm, stop_early, intermediates_only)
end

# `forward_intermediates` of the decoder `model` for the input embeddings `inputs_embeds`.
function decoder_intermediates(model::Qwen3_5TextModel, inputs_embeds, position_ids,
        attention_mask; indices, norm, stop_early, intermediates_only)
    take, max_index = feature_take_indices(length(model.layers), indices)
    x, position_embeddings, kv_lengths = decoder_inputs(model, inputs_embeds, position_ids,
        attention_mask)
    layers = [h -> layer(h, position_embeddings, kv_lengths) for layer in model.layers]
    outs = stage_outputs(layers, x, stop_early ? max_index : length(layers))
    intermediates = [norm ? model.norm(outs[i]) : outs[i] for i in take]
    intermediates_only && return intermediates
    return model.norm(last(outs)), intermediates
end

"""
    qwen3_5_text_model(; vocab_size = 248320, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `Qwen3_5TextModel`, the decoder of Qwen3.5, built from the fields of
its `Qwen3_5TextConfig` with the same names and defaults, and `full_attention_interval`,
`rope_theta`, `rope_scaling` and `partial_rotary_factor`, which the config reads from
config.json. It maps token indices of size `(seq_len, batch_size)`, or a vector, to the
hidden states after the final norm, and takes the keywords `attention_mask` and
`position_ids` (see [`qwen3_5_decode`](@ref Metalhead.Transformers.qwen3_5_decode), which
takes input embeddings instead).

Its layers have either a gated delta rule (`"linear_attention"`, `Qwen3_5GatedDeltaNet`) or
causal attention (`"full_attention"`), as `layer_types` says, by default every
`full_attention_interval`-th layer. The attention has a sigmoid gate on its output, norms
its queries and keys per head and rotates the first `head_dim * partial_rotary_factor`
features of each with the multimodal rotary embedding of `rope_parameters["mrope_section"]`
(only the default `rope_type`). The RMSNorms scale by `1 + weight`.
"""
function qwen3_5_text_model(; vocab_size = 248_320, hidden_size = 4096,
        intermediate_size = 12_288, num_hidden_layers = 32, num_attention_heads = 16,
        num_key_value_heads = 4, hidden_act = "silu", max_position_embeddings = 32_768,
        initializer_range = 0.02, rms_norm_eps = 1e-6, rope_parameters = nothing,
        rope_theta = nothing, rope_scaling = nothing, partial_rotary_factor = nothing,
        attention_bias = false, attention_dropout = 0.0, head_dim = 256,
        linear_conv_kernel_dim = 4, linear_key_head_dim = 128, linear_value_head_dim = 128,
        linear_num_key_heads = 16, linear_num_value_heads = 32, layer_types = nothing,
        full_attention_interval = 4)
    types = qwen3_5_layer_types(; num_hidden_layers, layer_types, full_attention_interval)
    head_dim, num_key_value_heads = attention_heads(; hidden_size, num_attention_heads,
        num_key_value_heads, head_dim)
    linear_num_value_heads % linear_num_key_heads == 0 ||
        error("`linear_num_value_heads` ($linear_num_value_heads) must be a multiple of `linear_num_key_heads` ($linear_num_key_heads)")
    rotary_emb = qwen3_5_rotary_embedding(head_dim; rope_parameters, rope_theta,
        rope_scaling, partial_rotary_factor)
    act = act_fn(hidden_act)
    init = normal_init(initializer_range)
    dense(in_out, bias = false) = Dense(in_out; bias, init)
    norm(dim) = Gemma3RMSNorm(dim; eps = rms_norm_eps)  # Qwen3_5RMSNorm
    function attention()
        return Qwen3_5Attention(head_dim,
            dense(hidden_size => 2 * num_attention_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(hidden_size => num_key_value_heads * head_dim, attention_bias),
            dense(num_attention_heads * head_dim => hidden_size, attention_bias),
            norm(head_dim), norm(head_dim), dropout_or_identity(attention_dropout))
    end
    function linear_attention()
        key_dim = linear_key_head_dim * linear_num_key_heads
        value_dim = linear_value_head_dim * linear_num_value_heads
        conv_dim = 2 * key_dim + value_dim
        kernel = linear_conv_kernel_dim
        conv1d = Conv((kernel,), conv_dim => conv_dim; groups = conv_dim,
            pad = (kernel - 1, 0), bias = false, init)
        A = 0.01f0 .+ (16 - 0.01f0) .* rand(Float32, linear_num_value_heads)
        return Qwen3_5GatedDeltaNet(linear_num_key_heads, linear_num_value_heads,
            linear_key_head_dim, linear_value_head_dim, conv1d, act,
            ones(Float32, linear_num_value_heads), log.(A),
            Qwen3_5RMSNormGated(linear_value_head_dim; eps = rms_norm_eps),
            dense(value_dim => hidden_size), dense(hidden_size => conv_dim),
            dense(hidden_size => value_dim), dense(hidden_size => linear_num_value_heads),
            dense(hidden_size => linear_num_value_heads))
    end
    function decoder_layer(layer_type)
        linear = layer_type == "linear_attention"
        mlp = LlamaMLP(dense(hidden_size => intermediate_size),
            dense(hidden_size => intermediate_size),
            dense(intermediate_size => hidden_size), act)
        return Qwen3_5DecoderLayer(linear ? linear_attention() : nothing,
            linear ? nothing : attention(), mlp, norm(hidden_size), norm(hidden_size))
    end
    return Qwen3_5TextModel(Embedding(vocab_size => hidden_size; init),
        [decoder_layer(t) for t in types], norm(hidden_size), rotary_emb)
end

"""
    qwen3_5_for_causal_lm(; vocab_size = 248320, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `Qwen3_5ForCausalLM`, the text model of Qwen3.5: the decoder of
[`qwen3_5_text_model`](@ref Metalhead.Transformers.qwen3_5_text_model), with its keywords,
and the output projection `lm_head`, tied to the token embeddings with
`tie_word_embeddings`. It takes and returns the arrays of
[`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm), and the keyword
`position_ids` of [`qwen3_5_decode`](@ref Metalhead.Transformers.qwen3_5_decode).
The multi-token prediction weights (`mtp.*`) of the checkpoints are not loaded, as in
transformers.
"""
function qwen3_5_for_causal_lm(; vocab_size = 248_320, hidden_size = 4096,
        intermediate_size = 12_288, num_hidden_layers = 32, num_attention_heads = 16,
        num_key_value_heads = 4, hidden_act = "silu", max_position_embeddings = 32_768,
        initializer_range = 0.02, rms_norm_eps = 1e-6, tie_word_embeddings = false,
        rope_parameters = nothing, rope_theta = nothing, rope_scaling = nothing,
        partial_rotary_factor = nothing, attention_bias = false, attention_dropout = 0.0,
        head_dim = 256, linear_conv_kernel_dim = 4, linear_key_head_dim = 128,
        linear_value_head_dim = 128, linear_num_key_heads = 16, linear_num_value_heads = 32,
        layer_types = nothing, full_attention_interval = 4)
    model = qwen3_5_text_model(; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_act,
        max_position_embeddings, initializer_range, rms_norm_eps, rope_parameters,
        rope_theta, rope_scaling, partial_rotary_factor, attention_bias, attention_dropout,
        head_dim, linear_conv_kernel_dim, linear_key_head_dim, linear_value_head_dim,
        linear_num_key_heads, linear_num_value_heads, layer_types, full_attention_interval)
    lm_head = tie_word_embeddings ? TiedLMHead(model.embed_tokens.weight) :
              Dense(hidden_size => vocab_size; bias = false,
        init = normal_init(initializer_range))
    return Qwen3_5ForCausalLM(model, lm_head)
end

# `layer_types`, computed as `Qwen3_5TextConfig` does when the config has none: every
# `full_attention_interval`-th layer has full attention, the others linear attention.
function qwen3_5_layer_types(; num_hidden_layers, layer_types, full_attention_interval)
    types = something(layer_types,
        [(i + 1) % full_attention_interval == 0 ? "full_attention" : "linear_attention"
         for i in 0:(num_hidden_layers - 1)])
    length(types) == num_hidden_layers ||
        error("`layer_types` has $(length(types)) entries, `num_hidden_layers` is $num_hidden_layers")
    supported = ("linear_attention", "full_attention")
    for t in types
        t in supported ||
            error("`layer_types` entry $(repr(t)) is not supported; supported: $supported")
    end
    return types
end

# The rotary embedding of `rope_parameters` (or `rope_scaling`), whose `rope_theta` and
# `partial_rotary_factor` default to the keywords and then to 10000 and 0.25, as in
# `Qwen3_5TextConfig.convert_rope_params_to_dict`. transformers ignores `mrope_interleaved`:
# the sections are always interleaved.
function qwen3_5_rotary_embedding(head_dim; rope_parameters, rope_theta, rope_scaling,
        partial_rotary_factor)
    params = string_keys(something(rope_scaling, rope_parameters, (;)))
    rope_type = get(params, "rope_type", get(params, "type", "default"))
    rope_type == "default" ||
        error("`rope_type = $(repr(rope_type))` is not supported by the Metalhead port of Qwen3_5ForCausalLM; supported: \"default\"")
    theta = Float64(get(params, "rope_theta", something(rope_theta, 10_000.0)))
    factor = get(params, "partial_rotary_factor", something(partial_rotary_factor, 0.25))
    rotary_dim = trunc(Int, head_dim * factor)
    rotary_dim > 0 && iseven(rotary_dim) ||
        error("`head_dim * partial_rotary_factor` ($rotary_dim) must be even and positive")
    section = Tuple(Int.(collect(get(params, "mrope_section", (11, 11, 10)))))
    length(section) == 3 ||
        error("`mrope_section` must have 3 entries (temporal, height, width), got $section")
    return Qwen3_5TextRotaryEmbedding(rotary_dim, theta, section)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3_5ForCausalLM", qwen3_5_for_causal_lm;
    family = "qwen", ignore = [r"^mtp\."]))

# The vision-language model: the vision encoder of qwen3_5_vision.jl, whose merged tokens
# replace the image tokens of the text, and the text model.

@concrete struct Qwen3_5Model
    visual
    language_model
    image_token_id
end
Flux.@layer Qwen3_5Model

# The input embeddings of the decoder and their (3, seq_len, batch) positions: the
# embeddings of the tokens, with the images' merged tokens in place of their image tokens.
function multimodal_inputs(m::Qwen3_5Model, input_ids, pixel_values, image_grid_thw,
        attention_mask)
    x = m.language_model.embed_tokens(input_ids)
    pixel_values === nothing && return x, nothing
    image_grid_thw === nothing && error("`pixel_values` need their `image_grid_thw`")
    features = m.visual(pixel_values, image_grid_thw).pooler_output
    image_index = m.image_token_id + 1
    x = merge_image_features(x, input_ids, features, image_index)
    positions = rope_index(input_ids, image_grid_thw, image_index,
        m.visual.spatial_merge_size, attention_mask)
    return x, positions
end

"""
    (model::Qwen3_5Model)(input_ids; pixel_values = nothing, image_grid_thw = nothing,
                          attention_mask = nothing)

The hidden states after the final norm of the vision-language model of Qwen3.5. The token
indices `input_ids` (seq_len, batch_size), or a vector, hold one run of image tokens for
each image of `pixel_values` and `image_grid_thw` (see
[`qwen3_5_for_conditional_generation`](@ref
Metalhead.Transformers.qwen3_5_for_conditional_generation)).
"""
function (m::Qwen3_5Model)(input_ids::AbstractMatrix{<:Integer}; pixel_values = nothing,
        image_grid_thw = nothing, attention_mask = nothing)
    x, positions = multimodal_inputs(m, input_ids, pixel_values, image_grid_thw,
        attention_mask)
    return qwen3_5_decode(m.language_model, x, positions; attention_mask)
end

function (m::Qwen3_5Model)(input_ids::AbstractVector{<:Integer}; attention_mask = nothing,
        kwargs...)
    mask = attention_mask === nothing ? nothing : reshape(attention_mask, :, 1)
    y = m(reshape(input_ids, :, 1); attention_mask = mask, kwargs...)
    return dropdims(y; dims = 3)
end

# The 0-based (3, seq_len, batch) positions of the multimodal rotary embedding, transformers'
# `get_rope_index` for images. Text tokens count up on the three rows. The tokens of an
# image, its `(t, h ÷ merge, w ÷ merge)` grid of merged patches, take the positions of their
# grid from the current position, which then advances by the larger of the two sides.
# Padding (`attention_mask`, on the right) keeps the position 0.
function rope_index(input_ids, image_grid_thw, image_index, merge, attention_mask)
    ids = Array(input_ids)
    grids = Array(image_grid_thw)
    lengths = something(kv_lengths_of(attention_mask, input_ids), fill(size(ids, 1),
        size(ids, 2)))
    positions = zeros(Int, 3, size(ids)...)
    image = 0
    for b in axes(ids, 2)
        pos, l = 0, 1
        while l <= lengths[b]
            if ids[l, b] != image_index
                positions[:, l, b] .= pos
                pos, l = pos + 1, l + 1
                continue
            end
            image += 1
            image <= size(grids, 2) ||
                error("the token indices hold more images than `image_grid_thw` ($(size(grids, 2)))")
            t, h, w = grids[1, image], grids[2, image] ÷ merge, grids[3, image] ÷ merge
            stop = l + t * h * w - 1
            stop <= lengths[b] && all(==(image_index), view(ids, l:stop, b)) ||
                error("image $image needs $(t * h * w) image tokens from position $l of sequence $b")
            grid = Iterators.product(0:(w - 1), 0:(h - 1), 0:(t - 1))  # width fastest
            for (k, (wi, hi, ti)) in zip(l:stop, grid)
                positions[:, k, b] .= (pos + ti, pos + hi, pos + wi)
            end
            pos, l = pos + max(h, w), stop + 1
        end
    end
    image == size(grids, 2) ||
        error("`image_grid_thw` has $(size(grids, 2)) images, the token indices $image")
    return positions
end
ChainRulesCore.@non_differentiable rope_index(::Any...)

# The embeddings `x` (hidden, seq_len, batch) with the image features (hidden, n) in place
# of the image tokens, in order, as torch's `masked_scatter`: one gather.
function merge_image_features(x, input_ids, features, image_index)
    d, n, b = size(x)
    index = ChainRulesCore.ignore_derivatives() do
        is_image = vec(Array(input_ids) .== image_index)
        count(is_image) == size(features, 2) ||
            error("the token indices hold $(count(is_image)) image tokens, the images give $(size(features, 2))")
        index = collect(1:(n * b))
        index[is_image] .= n * b .+ (1:size(features, 2))
        device_indices(x, index)
    end
    vectors = hcat(reshape(x, d, :), convert_like(x, features))
    return reshape(NNlib.gather(vectors, index), d, n, b)
end

@concrete struct Qwen3_5ForConditionalGeneration <: CausalLM
    model
    lm_head
end
Flux.@layer Qwen3_5ForConditionalGeneration

function forward_features(m::Qwen3_5ForConditionalGeneration, input_ids; kwargs...)
    return m.model(input_ids; kwargs...)
end

function (m::Qwen3_5ForConditionalGeneration)(input_ids; kwargs...)
    return forward_head(m, forward_features(m, input_ids; kwargs...))
end

"""
    (model::Qwen3_5ForConditionalGeneration)(inputs::NamedTuple)
    (model::Qwen3_5Model)(inputs::NamedTuple)

The model on the output of a [`Qwen3VLProcessor`](@ref
Metalhead.Transformers.Qwen3VLProcessor): its `input_ids` and `attention_mask`, and
`pixel_values` and `image_grid_thw` if it has images; other keys (`mm_token_type_ids`) are
ignored. The `input_ids` of one text, a vector, give the logits (or hidden states) of size
`(vocab_size, seq_len)`, so that `logits[:, end]` are those of the next token. A batch may
be padded on either side, as for a [`CausalLM`](@ref Metalhead.Transformers.CausalLM): a
batch padded on the left is moved to the right and its outputs back, so that each text
gets the outputs it gets alone, as in transformers.
"""
(m::Qwen3_5ForConditionalGeneration)(inputs::NamedTuple) = qwen3_5_on_inputs(m, inputs)
(m::Qwen3_5Model)(inputs::NamedTuple) = qwen3_5_on_inputs(m, inputs)

function qwen3_5_on_inputs(m, inputs)
    check_inputs(m, inputs, (:input_ids,);
        hint = "pass `text` to the processor, with an image placeholder for each image")
    input_ids, attention_mask, restore = right_padded(inputs.input_ids,
        get(inputs, :attention_mask, nothing))
    return restore(m(input_ids; pixel_values = get(inputs, :pixel_values, nothing),
        image_grid_thw = get(inputs, :image_grid_thw, nothing), attention_mask))
end

function forward_head(m::Qwen3_5ForConditionalGeneration, x; pre_logits::Bool = false)
    return pre_logits ? x : m.lm_head(x)
end

function forward_intermediates(m::Qwen3_5ForConditionalGeneration,
        input_ids::AbstractMatrix{<:Integer}; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, intermediates_only::Bool = false, pixel_values = nothing,
        image_grid_thw = nothing, attention_mask = nothing)
    x, positions = multimodal_inputs(m.model, input_ids, pixel_values, image_grid_thw,
        attention_mask)
    return decoder_intermediates(m.model.language_model, x, positions, attention_mask;
        indices, norm, stop_early, intermediates_only)
end

# The entries of the NamedTuple `config` (a sub-config of config.json) that `build`
# declares as keyword arguments.
function config_args(build, config)
    declared = Set(k for m in methods(build) for k in Base.kwarg_decl(m))
    return (; (k => v for (k, v) in pairs(something(config, (;))) if k in declared)...)
end

"""
    qwen3_5_for_conditional_generation(; text_config, vision_config, image_token_id = 248056,
                                       tie_word_embeddings = false)

Port of transformers' `Qwen3_5ForConditionalGeneration`, the vision-language model of the
Qwen3.5 checkpoints, built from its `Qwen3_5Config`: the vision encoder
[`qwen3_5_vision_model`](@ref Metalhead.Transformers.qwen3_5_vision_model) of
`vision_config`, the decoder [`qwen3_5_text_model`](@ref
Metalhead.Transformers.qwen3_5_text_model) of `text_config` (both NamedTuples of their
config's fields; other keys are ignored) and the output projection `lm_head`, tied to the
token embeddings with `tie_word_embeddings`. Videos and the multi-token prediction weights
(`mtp.*`) of the checkpoints are not supported.

The model maps token indices of size `(seq_len, batch_size)`, or a vector, to logits of
size `(vocab_size, seq_len, batch_size)`, as [`llama_for_causal_lm`](@ref
Metalhead.Transformers.llama_for_causal_lm) does, and takes the keyword arguments
`attention_mask` and, for images, `pixel_values` and `image_grid_thw`, as made by the
image processor of a checkpoint ([`Qwen2VLImageProcessor`](@ref
Metalhead.Transformers.Qwen2VLImageProcessor)). The token indices then hold, for each image
in order, a run of `prod(image_grid_thw[:, i]) ÷ 4` image tokens (`image_token_id + 1`,
`<|image_pad|>`), between the tokens of `<|vision_start|>` and `<|vision_end|>`, as the chat
template of the checkpoints writes them. The merged tokens of the image replace the image
tokens, and the multimodal rotary embedding gives them the positions of their grid. The
processor of a checkpoint, a [`Qwen3VLProcessor`](@ref
Metalhead.Transformers.Qwen3VLProcessor), makes all of these from images and a prompt with a
placeholder for each image, and the model takes its output as it is:
`model(proc(; images, text))`.
"""
function qwen3_5_for_conditional_generation(; text_config = (;), vision_config = (;),
        image_token_id = 248_056, tie_word_embeddings = false)
    language_model = qwen3_5_text_model(; config_args(qwen3_5_text_model, text_config)...)
    visual = qwen3_5_vision_model(; config_args(qwen3_5_vision_model, vision_config)...)
    weight = language_model.embed_tokens.weight
    lm_head = tie_word_embeddings ? TiedLMHead(weight) :
              Dense(size(weight, 1) => size(weight, 2); bias = false,
        init = normal_init(get(something(text_config, (;)), :initializer_range, 0.02)))
    return Qwen3_5ForConditionalGeneration(
        Qwen3_5Model(visual, language_model, image_token_id), lm_head)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3_5ForConditionalGeneration",
    qwen3_5_for_conditional_generation;
    family = "qwen", ignore = [r"^mtp\."],
    references = ["Qwen/Qwen3.5-0.8B" => "2fc06364715b967f1860aea9cf38778875588b17"]))
