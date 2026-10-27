# Ported from diffusers.models.transformers.transformer_z_image (diffusers 0.42.0,
# Apache-2.0, Alibaba Z-Image Team and The HuggingFace Team): text-to-image, without the
# Omni mode (several images per prompt, SigLIP features), ControlNet residuals and LoRA.
# Validated against test/fixtures/transformer_z_image_* and
# test/references/Tongyi-MAI--Z-Image-Turbo--transformer.

# The width of the conditioning of the adaptive norms, at most, and the multiple to which
# diffusers pads the image and the caption tokens of each sample.
const ADALN_EMBED_DIM = 256
const SEQ_MULTI_OF = 32

seq_padding(n) = mod(-n, SEQ_MULTI_OF)

# diffusers' `TimestepEmbedder` of Z-Image: the sinusoidal embeddings of the timesteps,
# cosines first, computed in at least `Float32` (in `Float64` for a `Float64` model, as the
# fixtures do), then an MLP.
@concrete struct ZImageTimestepEmbedder
    mlp
    frequency_embedding_size
end
Flux.@layer ZImageTimestepEmbedder

function (m::ZImageTimestepEmbedder)(t)
    weight = m.mlp.layers[1].weight
    T = promote_type(Float32, eltype(weight))
    freq = timestep_embedding(T.(t), m.frequency_embedding_size; flip_sin_to_cos = true,
        downscale_freq_shift = 0)
    return m.mlp(convert_like(weight, freq))
end

@concrete struct ZImageFeedForward
    w1
    w2
    w3
end
Flux.@layer ZImageFeedForward

(m::ZImageFeedForward)(x) = m.w2(swish(m.w1(x)) .* m.w3(x))

# diffusers' `Attention` with the `ZSingleStreamAttnProcessor` of Z-Image: self-attention of
# tokens `(dim, seq_len, batch)`, with an RMSNorm over the features of each query and key
# head, and the rotary embedding of their positions.
@concrete struct ZImageAttention
    heads
    to_q
    to_k
    to_v
    norm_q
    norm_k
    to_out
end
Flux.@layer ZImageAttention

function z_image_attention(dim; heads, head_dim, qk_norm)
    inner = heads * head_dim
    linear(in_out) = Dense(in_out; bias = false)
    norm() = qk_norm ? RMSNorm(head_dim; eps = 1e-5) : nothing
    return ZImageAttention(heads, linear(dim => inner), linear(dim => inner),
        linear(dim => inner), norm(), norm(),
        linear_dropout(inner => dim, 0.0; bias = false))
end

# `kv_lengths` are the lengths of the samples of a padded batch, or `nothing`.
function (m::ZImageAttention)(x, kv_lengths, rotary)
    _, n, b = size(x)
    heads(y) = reshape(y, :, m.heads, n, b)
    q, k, v = heads(m.to_q(x)), heads(m.to_k(x)), heads(m.to_v(x))
    q = m.norm_q === nothing ? q : m.norm_q(q)
    k = m.norm_k === nothing ? k : m.norm_k(k)
    q, k = rotate_z_image(q, rotary), rotate_z_image(k, rotary)
    y = NNlib.scaled_dot_product_attention(q, k, v; kv_lengths)
    return m.to_out(reshape(y, :, n, b))
end

# The features of `x` rotated by the tables of `z_image_rotary_tables`, in their element
# type (at least `Float32`), as diffusers rotates them in float32.
function rotate_z_image(x, (cos, sin, positions))
    y = NNlib.apply_rotary_embedding(convert_like(cos, x), cos, sin; positions,
        interleaved = true)
    return convert_like(x, y)
end

# diffusers' `ZImageTransformerBlock`. Without `adaLN_modulation` (the blocks of the
# `context_refiner`), it isn't conditioned on the timestep.
@concrete struct ZImageTransformerBlock
    attention
    feed_forward
    attention_norm1
    ffn_norm1
    attention_norm2
    ffn_norm2
    adaLN_modulation
end
Flux.@layer ZImageTransformerBlock

function z_image_block(dim; heads, norm_eps, qk_norm, modulation)
    norm() = RMSNorm(dim; eps = norm_eps)
    hidden = floor(Int, dim / 3 * 8)
    ff = ZImageFeedForward(Dense(dim => hidden; bias = false),
        Dense(hidden => dim; bias = false), Dense(dim => hidden; bias = false))
    adaln = modulation ? Chain(Dense(min(dim, ADALN_EMBED_DIM) => 4dim)) : nothing
    return ZImageTransformerBlock(
        z_image_attention(dim; heads, head_dim = dim ÷ heads, qk_norm), ff, norm(), norm(),
        norm(), norm(), adaln)
end

function (m::ZImageTransformerBlock)(x, kv_lengths, rotary, adaln_input = nothing)
    if m.adaLN_modulation === nothing
        x = x .+ m.attention_norm2(m.attention(m.attention_norm1(x), kv_lengths, rotary))
        return x .+ m.ffn_norm2(m.feed_forward(m.ffn_norm1(x)))
    end
    mod = m.adaLN_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = modulations(mod, 4)
    attn = m.attention(m.attention_norm1(x) .* (1 .+ scale_msa), kv_lengths, rotary)
    x = x .+ tanh.(gate_msa) .* m.attention_norm2(attn)
    return x .+ tanh.(gate_mlp) .*
                m.ffn_norm2(m.feed_forward(m.ffn_norm1(x) .* (1 .+ scale_mlp)))
end

@concrete struct ZImageFinalLayer
    norm_final
    linear
    adaLN_modulation
end
Flux.@layer ZImageFinalLayer

function (m::ZImageFinalLayer)(x, c)
    scale = 1 .+ reshape(m.adaLN_modulation(c), :, 1, size(c, 2))
    return m.linear(m.norm_final(x) .* scale)
end

# diffusers' `RopeEmbedder`, which isn't a module: rotary embeddings of positions with 3
# coordinates (frame, height, width), the features of each head split among the axes.
@concrete struct ZImageRopeEmbedder
    theta
    axes_dims
    axes_lens
end
Flux.@layer ZImageRopeEmbedder

# The rotary angles, in Float64, of the position ids `ids` of size `(3, n)`, concatenated
# over the axes, with diffusers' frequencies `1 / theta^(2k / d)`.
function rope_angles(m::ZImageRopeEmbedder, ids::AbstractMatrix{<:Integer})
    for (a, len) in enumerate(m.axes_lens)
        all(i -> 0 <= i < len, view(ids, a, :)) ||
            error("position ids of axis $a reach $(maximum(view(ids, a, :))), past `axes_lens` ($(repr(m.axes_lens)))")
    end
    return reduce(vcat,
        [[1 / m.theta^(2(k - 1) / d) for k in 1:(d ÷ 2)] .* Float64.(ids[a:a, :])
         for (a, d) in enumerate(m.axes_dims)])
end

# The tables of `rotate_z_image` for the position ids `ids` of size `(3, seq_len, batch)`:
# the cosines and sines of their angles in at least `Float32` (diffusers rounds the angles
# to float32), zero where `keep` (of size `(seq_len, batch)`) is false, one column for each
# token of each sample, and the positions of the tokens in them.
function rope_tables(m::ZImageRopeEmbedder, ids, keep, x)
    T = promote_type(Float32, eltype(x))
    n, b = size(keep)
    angles = T.(rope_angles(m, reshape(ids, 3, :)))
    mask = reshape(keep, 1, :)
    on_device(a) = copyto!(similar(x, T, size(a)), a)
    positions = device_indices(x, (1:n) .+ n .* (0:(b - 1))')
    return on_device(cos.(angles) .* mask), on_device(sin.(angles) .* mask), positions
end

# The rotary tables of the image tokens, of the caption tokens and of the sequence of both,
# as diffusers' `patchify_and_embed`, `_prepare_sequence` and `_build_unified_sequence` make
# them. The image tokens of a sample are at frames from `cap_len + 1`, for the padded length
# `cap_len` of its caption, their padding at (0, 0, 0), and the caption tokens at frames
# from 1. The rest is padding of the batch, to which diffusers gives no rotation (zero
# tables); its outputs are discarded. The caption of a sample has the position ids of
# `cap_len` tokens and then those of its padding, which diffusers' padding of the batch
# then cuts.
function z_image_rotary_tables(rope, (wt, ht, ft), nx, lengths, cap_lens, x)
    return ChainRulesCore.ignore_derivatives() do
        b, pmax = length(cap_lens), maximum(cap_lens)
        n = wt * ht * ft
        grid = reduce(hcat,
            vec([[f, h, w] for w in 0:(wt - 1), h in 0:(ht - 1), f in 0:(ft - 1)]))
        ids_x, ids_c, ids_u = zeros(Int, 3, nx, b), zeros(Int, 3, pmax, b),
        zeros(Int, 3, nx + pmax, b)
        keep_c, keep_u = falses(pmax, b), falses(nx + pmax, b)
        for i in 1:b
            p, l = cap_lens[i], lengths[i]
            ids_x[:, 1:n, i] .= grid .+ [p + 1, 0, 0]
            ids_c[1, 1:p, i] .= 1:p
            keep_c[1:min(2p - l, pmax), i] .= true
            ids_u[:, 1:nx, i] .= view(ids_x, :, :, i)
            ids_u[1, (nx + 1):(nx + p), i] .= 1:p
            keep_u[1:(nx + p), i] .= true
        end
        return rope_tables(rope, ids_x, trues(nx, b), x),
        rope_tables(rope, ids_c, keep_c, x), rope_tables(rope, ids_u, keep_u, x)
    end
end

# The patches of `pf` frames of `p × p` pixels of images `(width, height, frames, channels,
# batch)`, as tokens `(channels * p * p * pf, n, batch)` in diffusers' order (width
# fastest), and the size of their grid; and back.
function z_image_patchify(x, p, pf)
    w, h, f, c, b = size(x)
    w % p == 0 && h % p == 0 && f % pf == 0 ||
        error("images of size $((w, h, f)) (width, height, frames) don't divide into patches of $((p, p, pf))")
    wt, ht, ft = w ÷ p, h ÷ p, f ÷ pf
    y = permutedims(reshape(x, p, wt, p, ht, pf, ft, c, b), (7, 1, 3, 5, 2, 4, 6, 8))
    return reshape(y, c * p * p * pf, wt * ht * ft, b), (wt, ht, ft)
end

function z_image_unpatchify(y, (wt, ht, ft), p, pf, c)
    z = reshape(y, c, p, p, pf, wt, ht, ft, size(y, 3))
    return reshape(permutedims(z, (2, 5, 3, 6, 4, 7, 1, 8)), p * wt, p * ht, pf * ft, c, :)
end

zeros_like(x, dims...) = ChainRulesCore.ignore_derivatives(() -> fill!(similar(x, dims), 0))

# A caption sequence of the batch: the embedded tokens of a sample, the pad token up to the
# padded length of its caption, and zeros (the padding of the batch) up to `len`.
function caption_tokens(tokens, pad_token, cap_len, len)
    d, l = size(tokens)
    return hcat(tokens, repeat(pad_token, 1, cap_len - l),
        zeros_like(tokens, d, len - cap_len))
end

caption_list(c::AbstractVector{<:AbstractMatrix}) = c
caption_list(c::AbstractArray{<:Any, 3}) = [c[:, :, i] for i in axes(c, 3)]

@concrete struct ZImageTransformer2DModel
    in_channels
    t_scale
    all_x_embedder
    all_final_layer
    noise_refiner
    context_refiner
    t_embedder
    cap_embedder
    x_pad_token
    cap_pad_token
    layers
    rope_embedder
end
Flux.@layer ZImageTransformer2DModel

"""
    (transformer::ZImageTransformer2DModel)(x, t, cap_feats; patch_size = 2,
                                            f_patch_size = 1)

The output of the Z-Image transformer for the latent images `x`, of size
`(width, height, in_channels, batch)` or `(width, height, frames, in_channels, batch)`, at
the timesteps `t` (a vector of `batch` values in [0, 1]: 1 - σ for the noise level σ),
conditioned on the captions `cap_feats`: a vector of `batch` matrices of size
`(cap_feat_dim, length)`, the prompt embeddings of each image without their padding, or an
array of size `(cap_feat_dim, length, batch)` when they have the same length. Returns an
array of the size of `x`, the opposite of the velocity of the flow: Z-Image pipelines step
along its negation.

As in diffusers, the patches of each image and the tokens of its caption are padded to a
multiple of 32 with the learned `x_pad_token` and `cap_pad_token`, and the captions of a
batch are then padded to the same length, which the attention masks. `patch_size` and
`f_patch_size` select one of the pairs of `all_patch_size` and `all_f_patch_size`.
"""
function (m::ZImageTransformer2DModel)(x, t, cap_feats; patch_size::Integer = 2,
        f_patch_size::Integer = 1)
    key = Symbol(patch_size, "-", f_patch_size)
    haskey(m.all_x_embedder, key) ||
        error("the transformer has no patch embedding for `patch_size = $patch_size` and `f_patch_size = $f_patch_size`, only for $(join(keys(m.all_x_embedder), ", ")) (`all_patch_size`-`all_f_patch_size`)")
    images = ndims(x) == 4 ?
             reshape(x, size(x, 1), size(x, 2), 1, size(x, 3), size(x, 4)) : x
    ndims(images) == 5 ||
        error("`x` must have size (width, height, channels, batch) or (width, height, frames, channels, batch), not $(size(x))")
    caps = caption_list(cap_feats)
    b = size(images, 5)
    length(caps) == length(t) == b ||
        error("got $b images, $(length(t)) timesteps and $(length(caps)) captions")
    adaln = convert_like(x, m.t_embedder(t .* eltype(t)(m.t_scale)))

    # diffusers pads the patches with the last one and the captions with their last token,
    # embeds them, and then replaces the padding by the pad tokens
    tokens, grid = z_image_patchify(images, patch_size, f_patch_size)
    n = size(tokens, 2)
    nx = n + seq_padding(n)
    padded = cat(tokens, repeat(tokens[:, n:n, :], 1, nx - n, 1); dims = 2)
    embedded = reshape(m.all_x_embedder[key](reshape(padded, size(padded, 1), :)), :, nx, b)
    hx = cat(embedded[:, 1:n, :], repeat(m.x_pad_token, 1, nx - n, b); dims = 2)

    lengths, cap_lens, starts = ChainRulesCore.ignore_derivatives() do
        lengths = [size(c, 2) for c in caps]
        cap_lens = lengths .+ seq_padding.(lengths)
        return lengths, cap_lens, cumsum([0; cap_lens[1:(end - 1)]])
    end
    pmax = maximum(cap_lens)
    raw = reduce(hcat,
        [hcat(c, repeat(c[:, end:end], 1, p - l))
         for (c, l, p) in zip(caps, lengths, cap_lens)])
    embedded_caps = m.cap_embedder(raw)
    hc = cat([caption_tokens(embedded_caps[:, s .+ (1:l)], m.cap_pad_token, p, pmax)
              for (s, l, p) in zip(starts, lengths, cap_lens)]...; dims = 3)

    rot_x, rot_c, rot_u = z_image_rotary_tables(m.rope_embedder, grid, nx, lengths,
        cap_lens, hx)
    # the lengths of the samples, which the attention masks, unless they are the same
    uniform = allequal(cap_lens)
    kv_c, kv_u = ChainRulesCore.ignore_derivatives() do
        return uniform ? (nothing, nothing) : (cap_lens, nx .+ cap_lens)
    end
    for layer in m.noise_refiner
        hx = layer(hx, nothing, rot_x, adaln)
    end
    for layer in m.context_refiner
        hc = layer(hc, kv_c, rot_c)
    end
    h = uniform ? cat(hx, hc; dims = 2) :
        cat([hcat(hx[:, :, i], caption_tokens(hc[:, 1:p, i], m.cap_pad_token, p, pmax))
             for (i, p) in enumerate(cap_lens)]...; dims = 3)
    for layer in m.layers
        h = layer(h, kv_u, rot_u, adaln)
    end
    y = m.all_final_layer[key](h, adaln)
    out = z_image_unpatchify(y[:, 1:n, :], grid, patch_size, f_patch_size, size(images, 4))
    return ndims(x) == 4 ? reshape(out, size(x)) : out
end

"""
    z_image_transformer_2d_model(; dim = 3840, n_layers = 30, n_heads = 30,
                                 cap_feat_dim = 2560, kwargs...)

Port of diffusers' `ZImageTransformer2DModel`, the transformer of Z-Image, built from the
arguments of its constructor (the `config.json` of the `transformer` of a pipeline), with
the same names and defaults, which are those of Z-Image (Turbo has
`axes_lens = (1536, 512, 512)`).
As in diffusers, `n_kv_heads` is unused. The Omni variant (`siglip_feat_dim`) is not
supported.

See [`ZImageTransformer2DModel`](@ref Metalhead.Diffusers.ZImageTransformer2DModel) for its
inputs and output.
"""
function z_image_transformer_2d_model(; all_patch_size = (2,), all_f_patch_size = (1,),
        in_channels = 16, dim = 3840, n_layers = 30, n_refiner_layers = 2, n_heads = 30,
        n_kv_heads = 30, norm_eps = 1e-5, qk_norm = true, cap_feat_dim = 2560,
        siglip_feat_dim = nothing, rope_theta = 256.0, t_scale = 1000.0,
        axes_dims = (32, 48, 48), axes_lens = (1024, 512, 512))
    siglip_feat_dim === nothing ||
        error("`siglip_feat_dim` (Z-Image Omni) is not supported by the Metalhead port")
    length(all_patch_size) == length(all_f_patch_size) ||
        error("`all_patch_size` and `all_f_patch_size` must have the same length")
    length(axes_dims) == length(axes_lens) == 3 ||
        error("`axes_dims` and `axes_lens` must have 3 entries (frame, height, width)")
    sum(axes_dims) == dim ÷ n_heads ||
        error("`axes_dims` ($(repr(axes_dims))) must add up to the head dimension ($(dim ÷ n_heads))")
    all(iseven, axes_dims) || error("`axes_dims` must be even")
    pairs = [Symbol(p, "-", f) for (p, f) in zip(all_patch_size, all_f_patch_size)]
    patches = [(p, f) for (p, f) in zip(all_patch_size, all_f_patch_size)]
    adaln_dim = min(dim, ADALN_EMBED_DIM)
    final_layer(p, f) = ZImageFinalLayer(LayerNorm(dim; affine = false, eps = 1e-6),
        Dense(dim => p * p * f * in_channels), Chain(swish, Dense(adaln_dim => dim)))
    block(modulation) = z_image_block(dim; heads = n_heads, norm_eps, qk_norm, modulation)
    t_embedder = ZImageTimestepEmbedder(
        Chain(Dense(ADALN_EMBED_DIM => 1024), swish, Dense(1024 => adaln_dim)),
        ADALN_EMBED_DIM)
    return ZImageTransformer2DModel(in_channels, Float64(t_scale),
        NamedTuple{Tuple(pairs)}(Tuple(Dense(f * p * p * in_channels => dim)
        for (p, f) in patches)),
        NamedTuple{Tuple(pairs)}(Tuple(final_layer(p, f) for (p, f) in patches)),
        [block(true) for _ in 1:n_refiner_layers],
        [block(false) for _ in 1:n_refiner_layers], t_embedder,
        Chain(RMSNorm(cap_feat_dim; eps = norm_eps), Dense(cap_feat_dim => dim)),
        zeros(Float32, dim, 1), zeros(Float32, dim, 1), [block(true) for _ in 1:n_layers],
        ZImageRopeEmbedder(Float64(rope_theta), Tuple(axes_dims), Tuple(axes_lens)))
end

register!(ArchSpec(DIFFUSERS, "ZImageTransformer2DModel", z_image_transformer_2d_model;
    family = "z_image",
    references = ["Tongyi-MAI/Z-Image-Turbo/transformer" => "f332072aa78be7aecdf3ee76d5c247082da564a6"]))
