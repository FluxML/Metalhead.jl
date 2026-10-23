# Ported from diffusers.models.transformers.transformer_flux2 (diffusers 0.42.0, Apache-2.0,
# The HuggingFace Team), without the KV cache of reference images (`kv_cache_mode`), LoRA and
# the fused projections. Validated against test/fixtures/transformer_flux2_* and
# test/references/black-forest-labs--FLUX.2-klein-4B--transformer.

# diffusers' `Flux2SwiGLU`: the SiLU of the first half of the features times the second half.
function swiglu(x)
    half = size(x, 1) ÷ 2
    return swish(x[1:half, :, :]) .* x[(half + 1):end, :, :]
end

@concrete struct Flux2FeedForward
    linear_in
    act_fn
    linear_out
end
Flux.@layer Flux2FeedForward

function flux2_feed_forward(dim; mult, bias)
    inner = floor(Int, dim * mult)
    return Flux2FeedForward(Dense(dim => 2inner; bias), swiglu, Dense(inner => dim; bias))
end

(m::Flux2FeedForward)(x) = m.linear_out(m.act_fn(m.linear_in(x)))

# The attention of the double-stream blocks: the image tokens and the text tokens have their
# own projections, and attend to each other jointly (text first).
@concrete struct Flux2Attention
    head_dim
    to_q
    to_k
    to_v
    norm_q
    norm_k
    to_out
    norm_added_q
    norm_added_k
    add_q_proj
    add_k_proj
    add_v_proj
    to_add_out
end
Flux.@layer Flux2Attention

function flux2_attention(dim; heads, head_dim, eps, bias)
    inner = heads * head_dim
    linear(in_out) = Dense(in_out; bias)
    norm() = RMSNorm(head_dim; eps)
    return Flux2Attention(head_dim, linear(dim => inner), linear(dim => inner),
        linear(dim => inner), norm(), norm(), linear_dropout(inner => dim, 0.0; bias), norm(),
        norm(), linear(dim => inner), linear(dim => inner), linear(dim => inner),
        linear(inner => dim))
end

function (m::Flux2Attention)(x, context, (cos, sin))
    n, nc, b = size(x, 2), size(context, 2), size(x, 3)
    heads(y) = reshape(y, m.head_dim, :, size(y, 2), b)
    q = cat(m.norm_added_q(heads(m.add_q_proj(context))), m.norm_q(heads(m.to_q(x))); dims = 3)
    k = cat(m.norm_added_k(heads(m.add_k_proj(context))), m.norm_k(heads(m.to_k(x))); dims = 3)
    v = cat(heads(m.add_v_proj(context)), heads(m.to_v(x)); dims = 3)
    q = NNlib.apply_rotary_embedding(q, cos, sin; interleaved = true)
    k = NNlib.apply_rotary_embedding(k, cos, sin; interleaved = true)
    y = reshape(NNlib.scaled_dot_product_attention(q, k, v), :, nc + n, b)
    return m.to_out(y[:, (nc + 1):end, :]), m.to_add_out(y[:, 1:nc, :])
end

# The attention of the single-stream blocks, fused with their feed-forward network: one
# projection gives the queries, keys, values and the input of the network, and one maps the
# attention and the network's hidden features back.
@concrete struct Flux2ParallelSelfAttention
    head_dim
    inner_dim
    to_qkv_mlp_proj
    mlp_act_fn
    norm_q
    norm_k
    to_out
end
Flux.@layer Flux2ParallelSelfAttention

function flux2_parallel_self_attention(dim; heads, head_dim, eps, bias, mlp_ratio)
    inner = heads * head_dim
    mlp_hidden = floor(Int, dim * mlp_ratio)
    return Flux2ParallelSelfAttention(head_dim, inner,
        Dense(dim => 3inner + 2mlp_hidden; bias), swiglu, RMSNorm(head_dim; eps),
        RMSNorm(head_dim; eps), Dense(inner + mlp_hidden => dim; bias))
end

function (m::Flux2ParallelSelfAttention)(x, (cos, sin))
    _, n, b = size(x)
    h = m.to_qkv_mlp_proj(x)
    d = m.inner_dim
    heads(y) = reshape(y, m.head_dim, :, n, b)
    q = m.norm_q(heads(h[1:d, :, :]))
    k = m.norm_k(heads(h[(d + 1):2d, :, :]))
    v = heads(h[(2d + 1):3d, :, :])
    q = NNlib.apply_rotary_embedding(q, cos, sin; interleaved = true)
    k = NNlib.apply_rotary_embedding(k, cos, sin; interleaved = true)
    y = reshape(NNlib.scaled_dot_product_attention(q, k, v), :, n, b)
    return m.to_out(vcat(y, m.mlp_act_fn(h[(3d + 1):end, :, :])))
end

# The shift, scale and gate of each of `n ÷ 3` modulations, as `(dim, 1, batch)` arrays.
function modulations(mod, n)
    d = size(mod, 1) ÷ n
    return ntuple(i -> reshape(mod[((i - 1) * d + 1):(i * d), :], d, 1, :), n)
end

modulate(x, shift, scale) = (1 .+ scale) .* x .+ shift

@concrete struct Flux2SingleTransformerBlock
    norm
    attn
end
Flux.@layer Flux2SingleTransformerBlock

function (m::Flux2SingleTransformerBlock)(x, temb_mod, rotary)
    shift, scale, gate = modulations(temb_mod, 3)
    x = x .+ gate .* m.attn(modulate(m.norm(x), shift, scale), rotary)
    return clip_float16(x)
end

@concrete struct Flux2TransformerBlock
    norm1
    norm1_context
    attn
    norm2
    ff
    norm2_context
    ff_context
end
Flux.@layer Flux2TransformerBlock

# Returns the text tokens and the image tokens, in this order, as diffusers does.
function (m::Flux2TransformerBlock)(x, context, temb_mod_img, temb_mod_txt, rotary)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulations(temb_mod_img, 6)
    c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = modulations(
        temb_mod_txt, 6)
    attn, context_attn = m.attn(modulate(m.norm1(x), shift_msa, scale_msa),
        modulate(m.norm1_context(context), c_shift_msa, c_scale_msa), rotary)
    x = x .+ gate_msa .* attn
    x = x .+ gate_mlp .* m.ff(modulate(m.norm2(x), shift_mlp, scale_mlp))
    context = context .+ c_gate_msa .* context_attn
    context = context .+
              c_gate_mlp .* m.ff_context(modulate(m.norm2_context(context), c_shift_mlp,
                  c_scale_mlp))
    return clip_float16(context), x
end

# diffusers' `Flux2PosEmbed`: rotary embeddings of positions with several coordinates (time,
# height, width and layer for FLUX.2), the features of each head split among the axes.
@concrete struct Flux2PosEmbed
    theta
    axes_dim
end
Flux.@layer Flux2PosEmbed

# The tables of `NNlib.apply_rotary_embedding(...; interleaved = true)` of the positions `ids`
# of size `(n_axes, seq_len)` for the `pos_embed` of a transformer, one column for each token:
# the angles of the frequencies of each axis, concatenated, as diffusers'
# `repeat_interleave_real` pairs the features. Computed in Float64 (as diffusers does) on the
# CPU.
function rotary_tables(m, ids, x)
    return ChainRulesCore.ignore_derivatives() do
        pos = Float64.(Array(ids))
        size(pos, 1) == length(m.axes_dim) ||
            error("the position ids have $(size(pos, 1)) coordinates; `axes_dims_rope` has $(length(m.axes_dim))")
        angles = reduce(vcat,
            [[m.theta^(-2(i - 1) / d) for i in 1:(d ÷ 2)] .* pos[a:a, :]
             for (a, d) in enumerate(m.axes_dim)])
        return to_device_of(x, cos.(angles)), to_device_of(x, sin.(angles))
    end
end

@concrete struct Flux2TimestepGuidanceEmbeddings
    time_proj
    timestep_embedder
    guidance_embedder
end
Flux.@layer Flux2TimestepGuidanceEmbeddings

function (m::Flux2TimestepGuidanceEmbeddings)(timestep, guidance)
    temb = m.timestep_embedder(convert_like(timestep, m.time_proj(timestep)))
    (guidance === nothing || m.guidance_embedder === nothing) && return temb
    return temb .+ m.guidance_embedder(convert_like(guidance, m.time_proj(guidance)))
end

# `y` in the element type of `x`, as torch's `y.to(x.dtype)`.
convert_like(x, y) = eltype(y) == eltype(x) ? y : eltype(x).(y)

@concrete struct Flux2Modulation
    linear
    act_fn
end
Flux.@layer Flux2Modulation

(m::Flux2Modulation)(temb) = m.linear(m.act_fn(temb))

@concrete struct Flux2Transformer2DModel
    pos_embed
    time_guidance_embed
    double_stream_modulation_img
    double_stream_modulation_txt
    single_stream_modulation
    x_embedder
    context_embedder
    transformer_blocks
    single_transformer_blocks
    norm_out
    proj_out
end
Flux.@layer Flux2Transformer2DModel

# diffusers takes the position ids of the first batch element, of size (n_axes, seq_len).
first_ids(ids) = ndims(ids) == 3 ? ids[:, :, 1] : ids

"""
    (transformer::Flux2Transformer2DModel)(hidden_states, encoder_hidden_states, timestep,
                                           img_ids, txt_ids, guidance = nothing)

The velocity predicted by the FLUX.2 transformer for the image tokens `hidden_states` of size
`(in_channels, image_seq_len, batch)`, conditioned on the text tokens `encoder_hidden_states`
of size `(joint_attention_dim, text_seq_len, batch)`, at the `timestep`s (a vector of `batch`
values in [0, 1], the noise levels σ). `img_ids` and `txt_ids` are the integer coordinates of
the tokens, of size `(4, seq_len)` (or `(4, seq_len, batch)`, of which the first is used), from
which the rotary embeddings are computed. `guidance` is the guidance scale of
guidance-distilled checkpoints (`guidance_embeds = true`). Returns an array of size
`(patch_size^2 * out_channels, image_seq_len, batch)`.
"""
function (m::Flux2Transformer2DModel)(hidden_states, encoder_hidden_states, timestep,
        img_ids, txt_ids, guidance = nothing)
    num_txt_tokens = size(encoder_hidden_states, 2)
    like_x(t) = ChainRulesCore.ignore_derivatives(() -> to_device_of(hidden_states, t)) .* 1000
    temb = m.time_guidance_embed(like_x(timestep),
        guidance === nothing ? nothing : like_x(guidance))
    mod_img = m.double_stream_modulation_img(temb)
    mod_txt = m.double_stream_modulation_txt(temb)
    mod_single = m.single_stream_modulation(temb)
    x = m.x_embedder(hidden_states)
    context = m.context_embedder(encoder_hidden_states)
    rotary = ChainRulesCore.ignore_derivatives() do
        txt_cos, txt_sin = rotary_tables(m.pos_embed, first_ids(txt_ids), x)
        img_cos, img_sin = rotary_tables(m.pos_embed, first_ids(img_ids), x)
        return hcat(txt_cos, img_cos), hcat(txt_sin, img_sin)
    end
    for block in m.transformer_blocks
        context, x = block(x, context, mod_img, mod_txt, rotary)
    end
    x = cat(context, x; dims = 2)
    for block in m.single_transformer_blocks
        x = block(x, mod_single, rotary)
    end
    x = m.norm_out(x[:, (num_txt_tokens + 1):end, :], temb)
    return m.proj_out(x)
end

"""
    flux2_transformer_2d_model(; num_layers = 8, num_single_layers = 48, num_attention_heads = 48,
                               attention_head_dim = 128, kwargs...)

Port of diffusers' `Flux2Transformer2DModel`, the transformer of FLUX.2, built from the
arguments of its constructor (the `config.json` of the `transformer` of a pipeline), with the
same names and defaults. The defaults are those of FLUX.2 [dev]; FLUX.2 [klein] 4B has 5 double-
and 20 single-stream blocks of 24 heads.

See [`Flux2Transformer2DModel`](@ref Metalhead.Diffusers.Flux2Transformer2DModel) for its inputs
and output.
"""
function flux2_transformer_2d_model(; patch_size = 1, in_channels = 128, out_channels = nothing,
        num_layers = 8, num_single_layers = 48, attention_head_dim = 128,
        num_attention_heads = 48, joint_attention_dim = 15360,
        timestep_guidance_channels = 256, mlp_ratio = 3.0, axes_dims_rope = (32, 32, 32, 32),
        rope_theta = 2000, eps = 1e-6, guidance_embeds = true)
    sum(axes_dims_rope) == attention_head_dim ||
        error("`axes_dims_rope` ($(repr(axes_dims_rope))) must add up to `attention_head_dim` ($attention_head_dim)")
    all(iseven, axes_dims_rope) || error("`axes_dims_rope` must be even")
    dim = num_attention_heads * attention_head_dim
    out_channels = something(out_channels, in_channels)
    kw = (; heads = num_attention_heads, head_dim = attention_head_dim, eps, bias = false)
    norm() = LayerNorm(dim; affine = false, eps)
    double_block() = Flux2TransformerBlock(norm(), norm(), flux2_attention(dim; kw...), norm(),
        flux2_feed_forward(dim; mult = mlp_ratio, bias = false), norm(),
        flux2_feed_forward(dim; mult = mlp_ratio, bias = false))
    single_block() = Flux2SingleTransformerBlock(norm(),
        flux2_parallel_self_attention(dim; kw..., mlp_ratio))
    embedder() = timestep_embedding_layer(timestep_guidance_channels, dim; bias = false)
    time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
        Timesteps(timestep_guidance_channels, true, 0, 1), embedder(),
        guidance_embeds ? embedder() : nothing)
    modulation(sets) = Flux2Modulation(Dense(dim => 3dim * sets; bias = false), swish)
    return Flux2Transformer2DModel(Flux2PosEmbed(Float64(rope_theta), Tuple(axes_dims_rope)),
        time_guidance_embed, modulation(2), modulation(2), modulation(1),
        Dense(in_channels => dim; bias = false),
        Dense(joint_attention_dim => dim; bias = false),
        [double_block() for _ in 1:num_layers], [single_block() for _ in 1:num_single_layers],
        AdaLayerNormContinuous(dim, dim; elementwise_affine = false, eps, bias = false),
        Dense(dim => patch_size^2 * out_channels; bias = false))
end

register!(ArchSpec(DIFFUSERS, "Flux2Transformer2DModel", flux2_transformer_2d_model;
    family = "flux2",
    references = ["black-forest-labs/FLUX.2-klein-4B/transformer" => "e7b7dc27f91deacad38e78976d1f2b499d76a294"]))
