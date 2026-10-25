# Ported from transformers.models.sam3.modeling_sam3 (transformers 5.20, Apache-2.0, Meta AI
# and The Hugging Face Team): `Sam3Model`, the detector of SAM 3, which segments the objects
# of an image that a text and box prompts describe. Validated against
# test/fixtures/sam3_*. The video model (`Sam3VideoModel`) is ported for its detector only
# (see `sam3_video_model`): its tracker and the tracker of images (`Sam3TrackerModel`) are
# not.

# A constant built on the CPU, `a`, on the device of `x`, in its own eltype. A `BitArray`
# is made dense first, which GPU arrays copy from without scalar indexing.
device_array(x, a) = copyto!(similar(x, eltype(a), size(a)), convert(Array, a))

@concrete struct Sam3MLP
    activation_fn
    fc1
    fc2
    dropout
end
Flux.@layer Sam3MLP

(m::Sam3MLP)(x) = m.fc2(m.activation_fn.(m.dropout(m.fc1(x))))

# Multi-head attention of `query` (hidden, q_len, batch) on `key` and `value`
# (hidden, kv_len, batch), with the additive `bias` and the boolean `mask` of
# `NNlib.scaled_dot_product_attention`, of size (kv_len, q_len, heads, batch) or
# broadcastable to it.
@concrete struct Sam3Attention
    num_heads
    q_proj
    k_proj
    v_proj
    o_proj
end
Flux.@layer Sam3Attention

function (m::Sam3Attention)(query, key, value, bias = nothing; mask = nothing)
    _, n, b = size(query)
    heads(proj, x) = reshape(proj(x), :, m.num_heads, size(x, 2), size(x, 3))
    q, k, v = heads(m.q_proj, query), heads(m.k_proj, key), heads(m.v_proj, value)
    y = NNlib.scaled_dot_product_attention(q, k, v, bias; mask)
    return m.o_proj(reshape(y, :, n, b))
end

# A key padding mask (kv_len, batch) of `true` for the valid keys, as the `mask` of
# `NNlib.scaled_dot_product_attention`, on the device of `x`.
function key_padding_mask(x, mask)
    mask === nothing && return nothing
    return ChainRulesCore.ignore_derivatives() do
        valid = Array(mask) .!= 0
        device_array(x, reshape(valid, size(valid, 1), 1, 1, size(valid, 2)))
    end
end

# ViT backbone

@concrete struct Sam3ViTPatchEmbeddings
    projection
end
Flux.@layer Sam3ViTPatchEmbeddings

# The patches (hidden, W * H, batch) of the images (W, H, channels, batch).
function (m::Sam3ViTPatchEmbeddings)(pixel_values)
    y = m.projection(convert_like(m.projection.weight, pixel_values))
    w, h, d, b = size(y)
    return permutedims(reshape(y, w * h, d, b), (2, 1, 3))
end

@concrete struct Sam3ViTEmbeddings
    patch_embeddings
    position_embeddings
    dropout
    patch_size
end
Flux.@layer Sam3ViTEmbeddings

# The learned position embeddings (hidden, P * P, 1) of the pretraining grid of P × P
# patches, tiled over the w × h patches (not interpolated), as `_tile_position_embeddings`.
function tile_position_embeddings(pos, w, h)
    d, n = size(pos, 1), size(pos, 2)
    p = isqrt(n)
    p == w == h && return pos
    tiled = repeat(reshape(pos, d, p, p), 1, w ÷ p + 1, h ÷ p + 1)[:, 1:w, 1:h]
    return reshape(tiled, d, w * h, 1)
end

function (m::Sam3ViTEmbeddings)(pixel_values)
    w, h = size(pixel_values, 1) ÷ m.patch_size, size(pixel_values, 2) ÷ m.patch_size
    x = m.patch_embeddings(pixel_values)
    return m.dropout(x .+ tile_position_embeddings(m.position_embeddings, w, h))
end

# The 2D axial rotary embedding of the ViT, with no state: `vit_rotary_tables` computes its
# tables.
@concrete struct Sam3ViTRotaryEmbedding
    head_dim
    rope_theta
end
Flux.@layer Sam3ViTRotaryEmbedding

# The tables of `NNlib.apply_rotary_embedding` with `interleaved = true` (transformers'
# `rotate_pairwise`) for a grid of `(gw, gh)` patches, width fastest, whose positions are
# scaled by `scale`: the first half of the pairs of features turn with the column of the
# patch, the second half with its row, at the same frequencies. Computed in Float64 and
# returned in at least Float32, as transformers rotates in Float32.
function vit_rotary_tables(rotary, grid, scale, x)
    return ChainRulesCore.ignore_derivatives() do
        gw, gh = grid
        spatial = rotary.head_dim ÷ 2
        freqs = 1 ./ (Float64(rotary.rope_theta) .^ ((0:2:(spatial - 1)) ./ spatial))
        columns = [(j * scale, i * scale) for j in 0:(gw - 1), i in 0:(gh - 1)]
        angles = [k <= length(freqs) ? first(c) * freqs[k] :
                  last(c) * freqs[k - length(freqs)]
                  for k in 1:(2 * length(freqs)), c in vec(columns)]
        T = promote_type(eltype(x), Float32)
        return device_array(x, T.(cos.(angles))), device_array(x, T.(sin.(angles)))
    end
end

@concrete struct Sam3ViTRoPEAttention
    num_heads
    q_proj
    k_proj
    v_proj
    o_proj
    attention_dropout
end
Flux.@layer Sam3ViTRoPEAttention

# Self-attention over the tokens (hidden, seq_len, windows) of each window.
function (m::Sam3ViTRoPEAttention)(x, position_embeddings)
    _, n, b = size(x)
    cos, sin = position_embeddings
    heads(proj) = reshape(proj(x), :, m.num_heads, n, b)
    rotate(a) = convert_like(a,
        NNlib.apply_rotary_embedding(a, cos, sin; interleaved = true))
    q, k = rotate(heads(m.q_proj)), rotate(heads(m.k_proj))
    y = NNlib.scaled_dot_product_attention(q, k, heads(m.v_proj);
        fdrop = attention_fdrop(m.attention_dropout, q))
    return m.o_proj(reshape(y, :, n, b))
end

# The tokens (hidden, W, H, batch) in windows of `ws × ws` (hidden, ws * ws, windows), after
# padding W and H with zeros to multiples of `ws`, as `window_partition`.
function window_partition(x, ws)
    d, w, h, b = size(x)
    pw, ph = cld(w, ws) * ws, cld(h, ws) * ws
    x = (pw, ph) == (w, h) ? x : NNlib.pad_zeros(x, (0, pw - w, 0, ph - h); dims = (2, 3))
    windows = permutedims(reshape(x, d, ws, pw ÷ ws, ws, ph ÷ ws, b), (1, 2, 4, 3, 5, 6))
    return reshape(windows, d, ws * ws, :), (pw, ph)
end

function window_unpartition(windows, ws, (pw, ph), (w, h))
    d = size(windows, 1)
    x = permutedims(reshape(windows, d, ws, ws, pw ÷ ws, ph ÷ ws, :), (1, 2, 4, 3, 5, 6))
    x = reshape(x, d, pw, ph, :)
    return (pw, ph) == (w, h) ? x : x[:, 1:w, 1:h, :]
end

# A layer of windowed attention (`window_size > 0`) or of global attention over the patches
# of an image of `image_size`, whose grid `grid` (width, height) sets the rotary positions.
@concrete struct Sam3ViTLayer
    layer_norm1
    rotary_emb
    attention
    layer_norm2
    mlp
    dropout
    window_size
    grid
    position_scale
end
Flux.@layer Sam3ViTLayer

function (m::Sam3ViTLayer)(x)
    d, w, h, b = size(x)
    y = m.layer_norm1(x)
    if m.window_size > 0
        y, padded = window_partition(y, m.window_size)
    else
        prod(m.grid) == w * h ||
            error("the global attention layers take $(m.grid[1]) × $(m.grid[2]) patches, the grid of `image_size`; got $w × $h")
        y = reshape(y, d, w * h, b)
    end
    y = m.attention(y, vit_rotary_tables(m.rotary_emb, m.grid, m.position_scale, y))
    y = m.window_size > 0 ? window_unpartition(y, m.window_size, padded, (w, h)) :
        reshape(y, d, w, h, b)
    x = x .+ y
    return x .+ m.dropout(m.mlp(m.layer_norm2(x)))
end

@concrete struct Sam3ViTModel
    embeddings
    layer_norm
    layers
    patch_size
end
Flux.@layer Sam3ViTModel

"""
    (model::Sam3ViTModel)(pixel_values)

The `last_hidden_state` (hidden, W * H, batch) of the ViT of SAM 3 on the images
`pixel_values` (width, height, channels, batch), whose W × H patches it holds, width
fastest.
"""
function (m::Sam3ViTModel)(pixel_values)
    w, h = size(pixel_values, 1) ÷ m.patch_size, size(pixel_values, 2) ÷ m.patch_size
    x = m.embeddings(pixel_values)
    d, _, b = size(x)
    x = m.layer_norm(reshape(x, d, w, h, b))
    for layer in m.layers
        x = layer(x)
    end
    return (; last_hidden_state = reshape(x, d, w * h, b))
end

# The vision encoder: the ViT and the feature pyramid of its neck

@concrete struct Sam3SinePositionEmbedding
    num_position_features
    temperature
    normalize
    scale
end
Flux.@layer Sam3SinePositionEmbedding

# The sine features (2 * n_half, size(x)...) of each value of `x`, `x * scale` divided by
# the frequencies of `m`: sines and cosines of the even and odd features, interleaved, as
# `encode_1d_positions` and `encode_boxes`.
function sine_features(m, x)
    F = m.num_position_features
    dim_t = ChainRulesCore.ignore_derivatives() do
        t = Float64(m.temperature) .^ (2 .* ((0:(F - 1)) .÷ 2) ./ F)
        device_array(x, reshape(eltype(x).(t[1:2:end]), :, ntuple(_ -> 1, ndims(x))...))
    end
    angles = reshape(x .* eltype(x)(m.scale), 1, size(x)...) ./ dim_t
    pairs = cat(reshape(sin.(angles), 1, size(angles)...),
        reshape(cos.(angles), 1, size(angles)...); dims = 1)
    return reshape(pairs, F, size(x)...)
end

# The position encoding (W, H, 2 * num_position_features, 1) of a feature map `x`
# (W, H, channels, batch), `build_sine_position_embedding` without a mask: the features of
# the row, then those of the column, of each position, counted from 1 and normalized to
# `scale` with `normalize`.
function sine_position_embedding(m, x)
    return ChainRulesCore.ignore_derivatives() do
        w, h = size(x, 1), size(x, 2)
        T = eltype(x)
        ys, xs = T.(1:h), T.(1:w)
        if m.normalize
            ys = ys ./ (ys[end] + T(1e-6)) .* T(m.scale)
            xs = xs ./ (xs[end] + T(1e-6)) .* T(m.scale)
        end
        unit = (; m.num_position_features, m.temperature, scale = 1)
        pos_y = device_array(x, reshape(sine_features(unit, ys), :, 1, h))
        pos_x = device_array(x, reshape(sine_features(unit, xs), :, w, 1))
        F = m.num_position_features
        pos = cat(repeat(pos_y, 1, w, 1), repeat(pos_x, 1, 1, h); dims = 1)  # (2F, W, H)
        reshape(permutedims(pos, (2, 3, 1)), w, h, 2F, 1)
    end
end

@concrete struct Sam3FPNLayer
    scale_layers
    proj1
    proj2
end
Flux.@layer Sam3FPNLayer

function (m::Sam3FPNLayer)(x)
    return m.proj2(m.proj1(m.scale_layers(convert_like(m.proj1.weight, x))))
end

@concrete struct Sam3VisionNeck
    position_encoding
    fpn_layers
end
Flux.@layer Sam3VisionNeck

function (m::Sam3VisionNeck)(x)
    fpn_hidden_states = [layer(x) for layer in m.fpn_layers]
    fpn_position_encoding = [sine_position_embedding(m.position_encoding, y)
                             for y in fpn_hidden_states]
    return fpn_hidden_states, fpn_position_encoding
end

@concrete struct Sam3VisionModel
    backbone
    neck
end
Flux.@layer Sam3VisionModel

"""
    (model::Sam3VisionModel)(pixel_values)

The vision encoder of SAM 3 on the images `pixel_values` (width, height, channels, batch):
a NamedTuple of the `last_hidden_state` of the ViT (hidden, W * H, batch), and the
`fpn_hidden_states` and `fpn_position_encoding` of its neck, vectors of feature maps
(W, H, fpn_hidden_size, batch) at the `scale_factors` of the patch grid, and of their
sine position encodings (W, H, fpn_hidden_size, 1).
"""
function (m::Sam3VisionModel)(pixel_values)
    x = m.backbone(pixel_values).last_hidden_state
    p = m.backbone.patch_size
    w, h = size(pixel_values, 1) ÷ p, size(pixel_values, 2) ÷ p
    spatial = permutedims(reshape(x, :, w, h, size(x, 3)), (2, 3, 1, 4))
    fpn_hidden_states, fpn_position_encoding = m.neck(spatial)
    return (; last_hidden_state = x, fpn_hidden_states, fpn_position_encoding)
end

# Boxes (4, ...) from (cx, cy, w, h) to (x1, y1, x2, y2)
function box_cxcywh_to_xyxy(b)
    cx, cy, w, h = (selectdim(b, 1, i:i) for i in 1:4)
    half = eltype(b)(0.5)
    return cat(cx .- half .* w, cy .- half .* h, cx .+ half .* w, cy .+ half .* h; dims = 1)
end

function inverse_sigmoid(x; eps = 1e-3)
    T = eltype(x)
    x = clamp.(x, zero(T), one(T))
    return log.(max.(x, T(eps)) ./ max.(one(T) .- x, T(eps)))
end

# Geometry encoder of the box prompts

# torchvision's `roi_align(x, boxes, roi_size)` with its defaults (`spatial_scale = 1`,
# `sampling_ratio = -1`, `aligned = false`) on the CPU, as a matrix (W * H, roi_size^2) of
# each box (x1, y1, x2, y2) in the pixels of a W × H map, computed in `T`: the average of
# the bilinear interpolations at a grid of `ceil(roi_side / roi_size)` points per bin and
# side, that are zero more than a pixel outside the map.
function roi_align_weights(box, w, h, roi, T)
    x1, y1, x2, y2 = T.(box)
    roi_w, roi_h = max(x2 - x1, one(T)), max(y2 - y1, one(T))
    bin_w, bin_h = roi_w / roi, roi_h / roi
    grid_w, grid_h = ceil(Int, roi_w / roi), ceil(Int, roi_h / roi)
    count = T(max(grid_w * grid_h, 1))
    weights = zeros(T, w * h, roi * roi)
    for ph in 0:(roi - 1), pw in 0:(roi - 1), iy in 0:(grid_h - 1), ix in 0:(grid_w - 1)
        y = y1 + ph * bin_h + (iy + T(0.5)) * bin_h / grid_h
        x = x1 + pw * bin_w + (ix + T(0.5)) * bin_w / grid_w
        (y < -1 || y > h || x < -1 || x > w) && continue
        y, x = max(y, zero(T)), max(x, zero(T))
        y_low, x_low = floor(Int, y), floor(Int, x)
        if y_low >= h - 1
            y_low = y_high = h - 1
            y = T(y_low)
        else
            y_high = y_low + 1
        end
        if x_low >= w - 1
            x_low = x_high = w - 1
            x = T(x_low)
        else
            x_high = x_low + 1
        end
        ly, lx = y - y_low, x - x_low
        hy, hx = 1 - ly, 1 - lx
        r = pw + roi * ph + 1
        for (xi, yi, wt) in ((x_low, y_low, hy * hx), (x_high, y_low, hy * lx),
            (x_low, y_high, ly * hx), (x_high, y_high, ly * lx))
            weights[xi + w * yi + 1, r] += wt / count
        end
    end
    return weights
end

# The features (roi_size, roi_size, channels, num_boxes * batch) of the boxes (4, num_boxes,
# batch) (x1, y1, x2, y2) on the maps `x` (W, H, channels, batch), box by box of each image.
function roi_align(x, boxes, roi)
    w, h, c, b = size(x)
    n = size(boxes, 2)
    weights = ChainRulesCore.ignore_derivatives() do
        T = promote_type(eltype(x), Float32)
        bs = Array(boxes)
        ws = [roi_align_weights(view(bs, :, i, j), w, h, roi, T) for i in 1:n, j in 1:b]
        device_array(x, eltype(x).(reshape(reduce(hcat, vec(ws)), w * h, :, b)))
    end
    y = NNlib.batched_mul(NNlib.batched_transpose(weights), reshape(x, w * h, c, b))
    y = permutedims(reshape(y, roi, roi, n, c, b), (1, 2, 4, 3, 5))
    return reshape(y, roi, roi, c, n * b)
end

# `concat_padded_sequences`: the valid tokens of `seq2` (hidden, n2, batch) right after those
# of `seq1` (hidden, n1, batch) in each sequence, both padded on the right as their masks
# (n, batch) say, in a sequence of n1 + n2 tokens whose padding is what transformers leaves
# there (`seq1`'s past its tokens, then zeros). Returns the sequence and its mask.
function concat_padded_sequences(seq1, mask1, seq2, mask2)
    d, n1, b = size(seq1)
    n2 = size(seq2, 2)
    index, mask = ChainRulesCore.ignore_derivatives() do
        len1 = vec(sum(Array(mask1) .!= 0; dims = 1))
        len2 = vec(sum(Array(mask2) .!= 0; dims = 1))
        zero_index = n1 * b + n2 * b + 1
        index = [len1[j] < p <= len1[j] + n2 ? n1 * b + (j - 1) * n2 + p - len1[j] :
                 p <= n1 ? (j - 1) * n1 + p : zero_index
                 for p in 1:(n1 + n2), j in 1:b]
        mask = [p <= len1[j] + len2[j] for p in 1:(n1 + n2), j in 1:b]
        device_indices(seq1, index), mask
    end
    padding = ChainRulesCore.ignore_derivatives(() -> fill!(similar(seq1, d, 1), 0))
    tokens = hcat(reshape(seq1, d, :), reshape(seq2, d, :), padding)
    return reshape(NNlib.gather(tokens, index), d, n1 + n2, b), mask
end

@concrete struct Sam3GeometryEncoderLayer
    layer_norm1
    self_attn
    dropout
    cross_attn
    layer_norm2
    mlp
    layer_norm3
end
Flux.@layer Sam3GeometryEncoderLayer

function (m::Sam3GeometryEncoderLayer)(prompt, vision, vision_pos, prompt_mask)
    y = m.layer_norm1(prompt)
    x = m.dropout(m.self_attn(y, y, y; mask = prompt_mask)) .+ prompt
    x = m.dropout(m.cross_attn(m.layer_norm2(x), vision .+ vision_pos, vision)) .+ x
    return m.dropout(m.mlp(m.layer_norm3(x))) .+ x
end

@concrete struct Sam3GeometryEncoder
    position_encoding
    label_embed
    cls_embed
    boxes_direct_project
    boxes_pool_project
    boxes_pos_enc_project
    vision_layer_norm
    final_proj
    prompt_layer_norm
    layers
    output_layer_norm
    roi_size
end
Flux.@layer Sam3GeometryEncoder

# `_encode_boxes`: the embeddings (hidden, num_boxes, batch) of the boxes (cx, cy, w, h)
# (4, num_boxes, batch), normalized to the map `vision` (W, H, hidden, batch), and their
# labels (num_boxes, batch), 0 or 1.
function encode_boxes(m::Sam3GeometryEncoder, boxes, labels, vision)
    w, h, d, b = size(vision)
    n = size(boxes, 2)
    embeds = m.boxes_direct_project(boxes)
    pixels = ChainRulesCore.ignore_derivatives() do
        box_cxcywh_to_xyxy(boxes) .* device_array(boxes, eltype(boxes)[w, h, w, h])
    end
    sampled = roi_align(vision, pixels, m.roi_size)
    embeds = embeds .+ reshape(m.boxes_pool_project(sampled), d, n, b)
    cx, cy, bw, bh = (boxes[i:i, :, :] for i in 1:4)
    features(c) = reshape(sine_features(m.position_encoding, c), :, n, b)
    pos = cat(features(cy), features(cx), bh, bw; dims = 1)
    embeds = embeds .+ m.boxes_pos_enc_project(pos)
    return m.label_embed(labels .+ 1) .+ embeds
end

"""
    (model::Sam3GeometryEncoder)(boxes, box_mask, box_labels, img_feats, img_pos_embeds)

The embeddings of the box prompts of SAM 3, with a learned token after the boxes of each
image: a NamedTuple of the `last_hidden_state` (hidden, num_boxes + 1, batch) and its
`attention_mask`. The boxes (4, num_boxes, batch) are (cx, cy, w, h) normalized to the
image, `box_mask` (num_boxes, batch) tells the valid ones (first in each image) and
`box_labels` their labels, 0 (negative) or 1 (positive). The boxes attend to the last
feature map of `img_feats` and its position encoding in `img_pos_embeds`.
"""
function (m::Sam3GeometryEncoder)(boxes, box_mask, box_labels, img_feats, img_pos_embeds)
    vision = img_feats[end]
    w, h, d, b = size(vision)
    vision_flat = permutedims(reshape(vision, w * h, d, b), (2, 1, 3))
    pos = img_pos_embeds[end]
    pos_flat = permutedims(reshape(pos, w * h, d, size(pos, 4)), (2, 1, 3))
    normalized = permutedims(m.vision_layer_norm(permutedims(vision, (3, 1, 2, 4))),
        (2, 3, 1, 4))
    boxes = convert_like(m.boxes_direct_project.weight, boxes)
    labels = ChainRulesCore.ignore_derivatives() do
        device_indices(vision, Array(box_labels))
    end
    prompt = encode_boxes(m, boxes, labels, normalized)
    cls = repeat(reshape(m.cls_embed.weight, d, 1, 1), 1, 1, b)
    prompt, prompt_mask = concat_padded_sequences(prompt, box_mask, cls, trues(1, b))
    prompt = m.prompt_layer_norm(m.final_proj(prompt))
    attention_mask = key_padding_mask(prompt, prompt_mask)
    for layer in m.layers
        prompt = layer(prompt, vision_flat, pos_flat, attention_mask)
    end
    return (; last_hidden_state = m.output_layer_norm(prompt), attention_mask = prompt_mask)
end

# DETR encoder and decoder

@concrete struct Sam3DetrEncoderLayer
    layer_norm1
    self_attn
    dropout
    cross_attn
    layer_norm2
    mlp
    layer_norm3
end
Flux.@layer Sam3DetrEncoderLayer

function (m::Sam3DetrEncoderLayer)(x, prompt, pos, prompt_mask)
    y = m.layer_norm1(x)
    q = y .+ pos
    x = m.dropout(m.self_attn(q, q, y)) .+ x
    x = m.dropout(m.cross_attn(m.layer_norm2(x), prompt, prompt; mask = prompt_mask)) .+ x
    return m.dropout(m.mlp(m.layer_norm3(x))) .+ x
end

@concrete struct Sam3DetrEncoder
    layers
end
Flux.@layer Sam3DetrEncoder

flatten_map(x) = permutedims(reshape(x, :, size(x, 3), size(x, 4)), (2, 1, 3))

"""
    (model::Sam3DetrEncoder)(vision_features, text_features, vision_pos_embeds, text_mask)

The DETR encoder of SAM 3: the tokens of the feature maps `vision_features` (a vector of
(W, H, hidden, batch)), flattened and concatenated, attend to each other with the position
encodings `vision_pos_embeds`, and to the prompt `text_features` (hidden, prompt_len, batch)
whose valid tokens `text_mask` (prompt_len, batch) tells (all without). Returns the
`last_hidden_state`, the flattened `pos_embeds_flattened`, the `text_features` and the
`spatial_shapes` `(height, width)` of the maps.
"""
function (m::Sam3DetrEncoder)(vision_features, text_features, vision_pos_embeds, text_mask)
    x = reduce(hcat, [flatten_map(f) for f in vision_features])
    pos = reduce(hcat, [flatten_map(p) for p in vision_pos_embeds])
    spatial_shapes = [(size(f, 2), size(f, 1)) for f in vision_features]
    mask = key_padding_mask(x, text_mask)
    for layer in m.layers
        x = layer(x, text_features, pos, mask)
    end
    return (; last_hidden_state = x, pos_embeds_flattened = pos, text_features,
        spatial_shapes)
end

@concrete struct Sam3DecoderMLP
    layer1
    layer2
    layer3
end
Flux.@layer Sam3DecoderMLP

function (m::Sam3DecoderMLP)(x)
    x = relu.(m.layer1(x))
    m.layer3 === nothing && return m.layer2(x)
    return m.layer3(relu.(m.layer2(x)))
end

@concrete struct Sam3DetrDecoderLayer
    self_attn
    self_attn_dropout
    self_attn_layer_norm
    text_cross_attn
    text_cross_attn_dropout
    text_cross_attn_layer_norm
    vision_cross_attn
    vision_cross_attn_dropout
    vision_cross_attn_layer_norm
    mlp
    mlp_layer_norm
    mlp_dropout
end
Flux.@layer Sam3DetrDecoderLayer

# The queries `x` (hidden, 1 + num_queries, batch), the presence token first, whose
# positions `query_pos` (hidden, num_queries, batch) don't include it.
function (m::Sam3DetrDecoderLayer)(x, query_pos, text, vision, vision_pos, text_mask,
        vision_bias)
    query_pos = NNlib.pad_zeros(query_pos, (1, 0); dims = 2)
    q = x .+ query_pos
    x = m.self_attn_layer_norm(x .+ m.self_attn_dropout(m.self_attn(q, q, x)))
    y = m.text_cross_attn(x .+ query_pos, text, text; mask = text_mask)
    x = m.text_cross_attn_layer_norm(x .+ m.text_cross_attn_dropout(y))
    y = m.vision_cross_attn(x .+ query_pos, vision .+ vision_pos, vision, vision_bias)
    x = m.vision_cross_attn_layer_norm(x .+ m.vision_cross_attn_dropout(y))
    return m.mlp_layer_norm(x .+ m.mlp_dropout(m.mlp(x)))
end

@concrete struct Sam3DetrDecoder
    layers
    output_layer_norm
    box_head
    query_embed
    reference_points
    presence_token
    presence_head
    presence_layer_norm
    ref_point_head
    box_rpb_embed_x
    box_rpb_embed_y
    position_encoding
    clamp_presence_logit_max_val
end
Flux.@layer Sam3DetrDecoder

# `_get_rpb_matrix`: the bias (H * W, num_queries, heads, batch) of the attention of the
# queries to the H × W tokens of the feature map, from the log-scaled distances of each
# row and column to the sides of the reference boxes (4, num_queries, batch).
function box_rpb_matrix(m::Sam3DetrDecoder, reference_boxes, (h, w))
    xyxy = box_cxcywh_to_xyxy(reference_boxes)
    _, q, b = size(xyxy)
    T = eltype(xyxy)
    coords_h, coords_w = ChainRulesCore.ignore_derivatives() do
        device_array(xyxy, T.(0:(h - 1)) ./ T(h)), device_array(xyxy, T.(0:(w - 1)) ./ T(w))
    end
    deltas_y = reshape(coords_h, 1, h, 1, 1) .- reshape(xyxy[[2, 4], :, :], 2, 1, q, b)
    deltas_x = reshape(coords_w, 1, w, 1, 1) .- reshape(xyxy[[1, 3], :, :], 2, 1, q, b)
    log_scale(d) = sign.(d .* 8) .* log2.(abs.(d .* 8) .+ 1) ./ 3
    embed_x = m.box_rpb_embed_x(log_scale(deltas_x))  # (heads, W, num_queries, batch)
    embed_y = m.box_rpb_embed_y(log_scale(deltas_y))
    heads = size(embed_x, 1)
    rpb = reshape(permutedims(embed_x, (2, 3, 1, 4)), w, 1, q, heads, b) .+
          reshape(permutedims(embed_y, (2, 3, 1, 4)), 1, h, q, heads, b)
    return reshape(rpb, w * h, q, heads, b)
end

batch_repeat(weight, b) = repeat(reshape(weight, size(weight)..., 1), 1, 1, b)

"""
    (model::Sam3DetrDecoder)(vision_features, text_features, vision_pos_encoding, text_mask,
                             spatial_shapes)

The DETR decoder of SAM 3, whose queries attend to the prompt `text_features` and to the
encoded `vision_features` (hidden, H * W, batch) of the feature map of size
`spatial_shapes`, refining a reference box at each layer. Returns the
`intermediate_hidden_states` (hidden, num_queries, batch, num_layers) of the queries after
each layer (normed), the `reference_boxes` (4, num_queries, batch, num_layers)
(cx, cy, w, h) they started from, and the `presence_logits` (1, batch, num_layers) of the
presence token.
"""
function (m::Sam3DetrDecoder)(vision_features, text_features, vision_pos_encoding,
        text_mask, spatial_shapes)
    b = size(vision_features, 3)
    query_embeds = batch_repeat(m.query_embed.weight, b)
    reference_boxes = sigmoid.(batch_repeat(m.reference_points.weight, b))
    x = cat(batch_repeat(m.presence_token.weight, b), query_embeds; dims = 2)
    mask = key_padding_mask(x, text_mask)
    hidden_states, boxes, presence_logits = (), (reference_boxes,), ()
    c = m.clamp_presence_logit_max_val
    for layer in m.layers
        query_pos = m.ref_point_head(reshape(sine_features(m.position_encoding,
                reference_boxes[[2, 1, 3, 4], :, :]), :, size(reference_boxes)[2:3]...))
        bias = length(spatial_shapes) == 1 ?
               NNlib.pad_zeros(box_rpb_matrix(m, reference_boxes, only(spatial_shapes)),
            (1, 0); dims = 2) : nothing
        x = layer(x, query_pos, text_features, vision_features, vision_pos_encoding, mask,
            bias)
        queries = x[:, 2:end, :]
        delta_boxes = m.box_head(m.output_layer_norm(queries))
        new_reference_boxes = sigmoid.(delta_boxes .+ inverse_sigmoid(reference_boxes))
        reference_boxes = ChainRulesCore.ignore_derivatives(new_reference_boxes)
        hidden_states = (hidden_states..., m.output_layer_norm(queries))
        boxes = (boxes..., new_reference_boxes)
        presence = m.presence_head(m.presence_layer_norm(x[:, 1:1, :]))
        presence_logits = (presence_logits...,
            clamp.(reshape(presence, 1, b), -eltype(presence)(c), eltype(presence)(c)))
    end
    stack_layers(xs) = cat(map(y -> reshape(y, size(y)..., 1), xs)...;
        dims = ndims(xs[1]) + 1)
    return (; intermediate_hidden_states = stack_layers(hidden_states),
        reference_boxes = stack_layers(boxes[1:(end - 1)]),
        presence_logits = stack_layers(presence_logits))
end

@concrete struct Sam3DotProductScoring
    text_mlp
    text_mlp_dropout
    text_mlp_out_norm
    text_proj
    query_proj
    scale
    clamp_max_val
end
Flux.@layer Sam3DotProductScoring

# The mean of the valid features (hidden, seq_len, batch) of each sequence: (hidden, batch).
function pool_text_features(x, text_mask)
    text_mask === nothing && return dropdims(sum(x; dims = 2); dims = 2) ./ size(x, 2)
    valid = ChainRulesCore.ignore_derivatives() do
        device_array(x, eltype(x).(reshape(Array(text_mask) .!= 0, 1, size(x)[2:3]...)))
    end
    count = max.(sum(valid; dims = 2), one(eltype(x)))
    return dropdims(sum(x .* valid; dims = 2) ./ count; dims = 2)
end

"""
    (model::Sam3DotProductScoring)(decoder_hidden_states, text_features, text_mask)

The classification logits (1, num_queries, batch, num_layers) of the queries
`decoder_hidden_states` (hidden, num_queries, batch, num_layers): their scaled dot products
with the pooled prompt `text_features`.
"""
function (m::Sam3DotProductScoring)(decoder_hidden_states, text_features, text_mask)
    text = m.text_mlp(text_features)
    text = m.text_mlp_out_norm(m.text_mlp_dropout(text) .+ text_features)
    proj_text = m.text_proj(pool_text_features(text, text_mask))
    proj_queries = m.query_proj(decoder_hidden_states)
    d, b = size(proj_text)
    T = eltype(proj_queries)
    scores = sum(proj_queries .* reshape(proj_text, d, 1, b, 1); dims = 1) .* T(m.scale)
    return clamp.(scores, -T(m.clamp_max_val), T(m.clamp_max_val))
end

# Mask decoder

@concrete struct Sam3MaskEmbedder
    layers
    activation
end
Flux.@layer Sam3MaskEmbedder

function (m::Sam3MaskEmbedder)(x)
    for (i, layer) in enumerate(m.layers)
        x = layer(x)
        i < length(m.layers) && (x = m.activation.(x))
    end
    return x
end

# torch's `F.interpolate(x, size, mode = "nearest")` of a map (W, H, channels, batch).
function interpolate_nearest(x, (w, h))
    (w, h) == size(x)[1:2] && return x
    w % size(x, 1) == 0 && h % size(x, 2) == 0 && return NNlib.upsample_nearest(x;
        size = (w, h))
    source(n, n_in) = floor.(Int, (0:(n - 1)) .* (n_in / n)) .+ 1
    iw, ih = ChainRulesCore.ignore_derivatives(() -> (source(w, size(x, 1)),
        source(h, size(x, 2))))
    return x[iw, ih, :, :]
end

@concrete struct Sam3PixelDecoder
    conv_layers
    norms
end
Flux.@layer Sam3PixelDecoder

# The finest of the feature maps (W, H, hidden, batch), fine to coarse, after merging the
# coarser ones into it, one by one.
function (m::Sam3PixelDecoder)(backbone_features)
    x = backbone_features[end]
    for (i, feature) in enumerate(reverse(backbone_features[1:(end - 1)]))
        x = interpolate_nearest(x, size(feature)[1:2]) .+ feature
        x = relu.(m.norms[i](m.conv_layers[i](x)))
    end
    return x
end

@concrete struct Sam3MaskDecoder
    pixel_decoder
    mask_embedder
    instance_projection
    semantic_projection
    prompt_cross_attn
    prompt_cross_attn_norm
    prompt_cross_attn_dropout
end
Flux.@layer Sam3MaskDecoder

"""
    (model::Sam3MaskDecoder)(decoder_queries, backbone_features, encoder_hidden_states,
                             prompt_features, prompt_mask)

The mask decoder of SAM 3: the `pred_masks` (W, H, num_queries, batch) of the queries
`decoder_queries` (hidden, num_queries, batch), the dot products of their embeddings with
those of the pixels of the finest map of `backbone_features`, whose coarsest map the
encoder output `encoder_hidden_states` replaces after attending to the prompt; and the
`semantic_seg` (W, H, 1, batch) of the pixels.
"""
function (m::Sam3MaskDecoder)(decoder_queries, backbone_features, encoder_hidden_states,
        prompt_features, prompt_mask)
    if prompt_features !== nothing
        normed = m.prompt_cross_attn_norm(encoder_hidden_states)
        y = m.prompt_cross_attn(normed, prompt_features, prompt_features;
            mask = key_padding_mask(normed, prompt_mask))
        encoder_hidden_states = encoder_hidden_states .+ m.prompt_cross_attn_dropout(y)
    end
    w, h, d, b = size(backbone_features[end])
    encoder_map = reshape(permutedims(encoder_hidden_states[:, 1:(w * h), :], (2, 1, 3)),
        w, h, d, b)
    pixel_embed = m.pixel_decoder([backbone_features[1:(end - 1)]; [encoder_map]])
    instance_embeds = m.instance_projection(pixel_embed)
    mask_embeddings = m.mask_embedder(decoder_queries)
    pw, ph = size(instance_embeds)[1:2]
    pred_masks = NNlib.batched_mul(reshape(instance_embeds, pw * ph, :, b), mask_embeddings)
    return (; pred_masks = reshape(pred_masks, pw, ph, :, b),
        semantic_seg = m.semantic_projection(pixel_embed))
end

# The model

@concrete struct Sam3Model
    vision_encoder
    text_encoder
    text_projection
    geometry_encoder
    detr_encoder
    detr_decoder
    mask_decoder
    dot_product_scoring
end
Flux.@layer Sam3Model

"""
    get_text_features(model::Sam3Model, input_ids; attention_mask = nothing)

The features of the text prompts of SAM 3: the `last_hidden_state` of its text encoder on
the token indices `input_ids` (seq_len, batch_size), 1-based and padded on the right after
the ones of `attention_mask`, and its projection to the hidden size of the detector
(`pooler_output`, as transformers names it). Pass them to the model in place of the token
indices to reuse them for several images.
"""
function get_text_features(m::Sam3Model, input_ids; attention_mask = nothing)
    y = m.text_encoder(input_ids; attention_mask)
    return (; y.last_hidden_state, pooler_output = m.text_projection(y.last_hidden_state))
end

"""
    get_vision_features(model::Sam3Model, pixel_values)

The output of the vision encoder of SAM 3 on the images `pixel_values` (see
[`Sam3VisionModel`](@ref Metalhead.Transformers.Sam3VisionModel)). Pass it to the model in
place of the images to reuse it for several prompts.
"""
get_vision_features(m::Sam3Model, pixel_values) = m.vision_encoder(pixel_values)

vision_outputs(m, x::NamedTuple) = x
vision_outputs(m, x) = get_vision_features(m, x)
text_outputs(m, x::NamedTuple; attention_mask) = x
text_outputs(m, x; attention_mask) = get_text_features(m, x; attention_mask)

# The prompt: the text features (hidden, text_len, batch), with those of the boxes after
# them, and its mask (prompt_len, batch) or `nothing` (all valid).
function sam3_prompt(m::Sam3Model, text_features, text_mask, fpn, fpn_pos, input_boxes,
        input_boxes_labels)
    (input_boxes === nothing || isempty(input_boxes)) && return text_features, text_mask
    n, b = size(input_boxes, 2), size(input_boxes, 3)
    labels, box_mask = ChainRulesCore.ignore_derivatives() do
        labels = input_boxes_labels === nothing ? ones(Int, n, b) :
                 Array(input_boxes_labels)
        size(labels) == (n, b) ||
            error("`input_boxes_labels` has size $(size(labels)), the boxes $((n, b))")
        ifelse.(labels .== -10, 0, labels), labels .!= -10
    end
    geometry = m.geometry_encoder(input_boxes, box_mask, labels, fpn, fpn_pos)
    features = geometry.last_hidden_state
    if size(text_features, 3) == 1 && size(features, 3) > 1
        text_features = repeat(text_features, 1, 1, size(features, 3))
        text_mask = text_mask === nothing ? nothing :
                    repeat(text_mask, 1, size(features, 3))
    end
    text_mask = ChainRulesCore.ignore_derivatives() do
        text_mask === nothing ? trues(size(text_features)[2:3]) : Array(text_mask) .!= 0
    end
    return cat(text_features, features; dims = 2), vcat(text_mask, geometry.attention_mask)
end

"""
    (model::Sam3Model)(pixel_values, input_ids; attention_mask = nothing,
                       input_boxes = nothing, input_boxes_labels = nothing)

Detect and segment the objects of the images `pixel_values` (width, height, channels,
batch) that the text prompts `input_ids` (seq_len, batch), 1-based token indices padded on
the right after the ones of `attention_mask`, describe; with `input_boxes`
(4, num_boxes, batch), boxes (cx, cy, w, h) normalized to the image, also those like the
boxes labelled 1 in `input_boxes_labels` (num_boxes, batch) and unlike those labelled 0,
-10 marking padding (by default, all are positive). The output of
[`get_vision_features`](@ref Metalhead.Transformers.get_vision_features) can replace the
images, and that of [`get_text_features`](@ref Metalhead.Transformers.get_text_features)
the token indices.

Returns a NamedTuple of
- `pred_masks` (W, H, num_queries, batch): the mask logits of the queries, at 4 times the
  resolution of the patch grid;
- `pred_boxes` (4, num_queries, batch): their boxes (x1, y1, x2, y2), normalized;
- `pred_logits` (num_queries, batch): their classification logits;
- `presence_logits` (1, batch): the logit that the prompt describes something in the image,
  whose sigmoid times that of `pred_logits` gives the scores of the queries;
- `semantic_seg` (W, H, 1, batch): the logits of a semantic segmentation of the prompt.
"""
function (m::Sam3Model)(pixel_values, input_ids; attention_mask = nothing,
        input_boxes = nothing, input_boxes_labels = nothing)
    vision = vision_outputs(m, pixel_values)
    fpn = vision.fpn_hidden_states[1:(end - 1)]
    fpn_pos = vision.fpn_position_encoding[1:(end - 1)]
    text_features = text_outputs(m, input_ids; attention_mask).pooler_output
    prompt, prompt_mask = sam3_prompt(m, text_features, attention_mask, fpn, fpn_pos,
        input_boxes, input_boxes_labels)
    encoder = m.detr_encoder([fpn[end]], prompt, [fpn_pos[end]], prompt_mask)
    decoder = m.detr_decoder(encoder.last_hidden_state, encoder.text_features,
        encoder.pos_embeds_flattened, prompt_mask, encoder.spatial_shapes)
    hidden_states = decoder.intermediate_hidden_states
    offsets = m.detr_decoder.box_head(hidden_states)
    all_pred_boxes = box_cxcywh_to_xyxy(sigmoid.(inverse_sigmoid(decoder.reference_boxes) .+
                                                 offsets))
    all_pred_logits = dropdims(m.dot_product_scoring(hidden_states, encoder.text_features,
            prompt_mask); dims = 1)
    masks = m.mask_decoder(hidden_states[:, :, :, end], fpn, encoder.last_hidden_state,
        prompt, prompt_mask)
    return (; masks.pred_masks, pred_boxes = all_pred_boxes[:, :, :, end],
        pred_logits = all_pred_logits[:, :, end],
        presence_logits = decoder.presence_logits[:, :, end], masks.semantic_seg)
end

# Builders

sam3_act(name) = act_fn(name; table = SAM3_ACT2FN)

function sam3_rope_theta(rope_parameters, rope_theta)
    p = something(rope_parameters, (;))
    get_param(k, default) = p isa AbstractDict ? get(p, String(k), default) :
                            get(p, k, default)
    rope_type = get_param(:rope_type, "axial")
    rope_type == "axial" ||
        error("`rope_type = $(repr(rope_type))` is not supported; SAM 3 supports only \"axial\"")
    return Float64(get_param(:rope_theta, something(rope_theta, 10_000.0)))
end

"""
    sam3_vit_model(; hidden_size = 1024, intermediate_size = 4736, num_hidden_layers = 32,
        num_attention_heads = 16, num_channels = 3, image_size = 1008, patch_size = 14,
        hidden_act = "gelu", layer_norm_eps = 1e-6, attention_dropout = 0.0,
        window_size = 24, global_attn_indexes = (7, 15, 23, 31),
        layer_scale_init_value = nothing, pretrain_image_size = 336, hidden_dropout = 0.0,
        initializer_range = 0.02, rope_parameters = nothing, rope_theta = nothing)

Port of transformers' `Sam3ViTModel`, the ViT of SAM 3, from the fields of its
`Sam3ViTConfig`: its layers attend within windows of `window_size × window_size` patches,
but for those of `global_attn_indexes` (0-based), which attend over the whole grid of an
image of `image_size`, the size the model takes. A 2D axial rotary embedding
(`rope_theta`, from `rope_parameters` too) gives the positions, scaled to those of a window
in the global layers; the learned position embeddings of the `pretrain_image_size` grid are
tiled over larger grids. As in transformers, `layer_scale_init_value` is not used.
"""
function sam3_vit_model(; hidden_size = 1024, intermediate_size = 4736,
        num_hidden_layers = 32, num_attention_heads = 16, num_channels = 3,
        image_size = 1008, patch_size = 14, hidden_act = "gelu", layer_norm_eps = 1e-6,
        attention_dropout = 0.0, window_size = 24, global_attn_indexes = (7, 15, 23, 31),
        layer_scale_init_value = nothing, pretrain_image_size = 336, hidden_dropout = 0.0,
        initializer_range = 0.02, rope_parameters = nothing, rope_theta = nothing)
    patch_size isa Integer ||
        error("`patch_size` must be an integer, as transformers' SAM 3 requires; got $patch_size")
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size = $hidden_size` must be divisible by `num_attention_heads = $num_attention_heads`")
    head_dim = hidden_size ÷ num_attention_heads
    head_dim % 4 == 0 ||
        error("the head dimension $head_dim must be divisible by 4 for the 2D rotary embedding")
    theta = sam3_rope_theta(rope_parameters, rope_theta)
    act = sam3_act(hidden_act)
    init = normal_init(initializer_range)
    dense(io) = Dense(io; init)
    norm() = LayerNorm(hidden_size; eps = Float32(layer_norm_eps))
    image_h, image_w = image_size isa Integer ? (image_size, image_size) : Tuple(image_size)
    pretrain = pretrain_image_size isa Integer ? pretrain_image_size :
               first(pretrain_image_size)
    # transformers' `precompute_positions` takes the grid as (width, height) from
    # `image_size[1]` and `image_size[0]`, which only square images tell apart
    image_grid = (image_h ÷ patch_size, image_w ÷ patch_size)
    layers = map(0:(num_hidden_layers - 1)) do i
        ws = i in global_attn_indexes ? 0 : window_size
        grid = ws > 0 ? (ws, ws) : image_grid
        attention = Sam3ViTRoPEAttention(num_attention_heads,
            dense(hidden_size => hidden_size), dense(hidden_size => hidden_size),
            dense(hidden_size => hidden_size), dense(hidden_size => hidden_size),
            dropout_or_identity(attention_dropout))
        mlp = Sam3MLP(act, dense(hidden_size => intermediate_size),
            dense(intermediate_size => hidden_size), dropout_or_identity(hidden_dropout))
        Sam3ViTLayer(norm(), Sam3ViTRotaryEmbedding(head_dim, theta), attention, norm(),
            mlp, dropout_or_identity(hidden_dropout), ws, grid, window_size / grid[1])
    end
    num_patches = (pretrain ÷ patch_size)^2
    embeddings = Sam3ViTEmbeddings(
        Sam3ViTPatchEmbeddings(Conv((patch_size, patch_size), num_channels => hidden_size;
            stride = patch_size, bias = false)),
        Float32(initializer_range) .* randn(Float32, hidden_size, num_patches, 1),
        dropout_or_identity(hidden_dropout), patch_size)
    return Sam3ViTModel(embeddings, norm(), layers, patch_size)
end

function sam3_fpn_layer(in_channels, fpn_dim, scale_factor)
    scale_layers = if scale_factor == 4
        Chain(ConvTranspose((2, 2), in_channels => in_channels ÷ 2; stride = 2), gelu_erf,
            ConvTranspose((2, 2), in_channels ÷ 2 => in_channels ÷ 4; stride = 2))
    elseif scale_factor == 2
        Chain(ConvTranspose((2, 2), in_channels => in_channels ÷ 2; stride = 2))
    elseif scale_factor == 1
        Chain(identity)  # torch's empty ModuleList; Flux's `Chain()` returns `nothing`
    elseif scale_factor == 0.5
        Chain(MaxPool((2, 2)))
    else
        error("`scale_factor = $scale_factor` is not supported; SAM 3 supports 4, 2, 1 and 0.5")
    end
    channels = scale_factor == 4 ? in_channels ÷ 4 :
               scale_factor == 2 ? in_channels ÷ 2 : in_channels
    return Sam3FPNLayer(scale_layers, Conv((1, 1), channels => fpn_dim),
        Conv((3, 3), fpn_dim => fpn_dim; pad = 1))
end

"""
    sam3_vision_model(; backbone_config = nothing, fpn_hidden_size = 256,
                      scale_factors = (4.0, 2.0, 1.0, 0.5))

Port of transformers' `Sam3VisionModel`, the vision encoder of SAM 3, from the fields of
its `Sam3VisionConfig`: the ViT [`sam3_vit_model`](@ref
Metalhead.Transformers.sam3_vit_model) of `backbone_config` (a NamedTuple of its config's
fields; other keys are ignored), and a neck that resamples its patch grid by each of
`scale_factors` into feature maps of `fpn_hidden_size` channels.
"""
function sam3_vision_model(; backbone_config = nothing, fpn_hidden_size = 256,
        scale_factors = (4.0, 2.0, 1.0, 0.5))
    backbone = sam3_vit_model(; config_args(sam3_vit_model, backbone_config)...)
    hidden_size = size(backbone.embeddings.position_embeddings, 1)
    position_encoding = Sam3SinePositionEmbedding(fpn_hidden_size ÷ 2, 10_000, true, 2π)
    fpn_layers = [sam3_fpn_layer(hidden_size, fpn_hidden_size, s) for s in scale_factors]
    return Sam3VisionModel(backbone, Sam3VisionNeck(position_encoding, fpn_layers))
end

# The fields of the configs of the geometry encoder, the DETR encoder and decoder and the
# mask decoder, with their defaults.
function sam3_block_config(config, defaults)
    given = something(config, (;))
    return merge(defaults,
        (; (k => v for (k, v) in pairs(given) if haskey(defaults, k))...))
end

const SAM3_ENCODER_DEFAULTS = (; hidden_size = 256, num_layers = 6, num_attention_heads = 8,
    intermediate_size = 2048, dropout = 0.1, hidden_act = "relu", hidden_dropout = 0.0,
    initializer_range = 0.02)

function sam3_attention(c, init)
    d = c.hidden_size
    d % c.num_attention_heads == 0 ||
        error("`hidden_size = $d` must be divisible by `num_attention_heads = $(c.num_attention_heads)`")
    dense() = Dense(d => d; init)
    return Sam3Attention(c.num_attention_heads, dense(), dense(), dense(), dense())
end

function sam3_mlp(c, init)
    return Sam3MLP(sam3_act(c.hidden_act),
        Dense(c.hidden_size => c.intermediate_size; init),
        Dense(c.intermediate_size => c.hidden_size; init),
        dropout_or_identity(c.hidden_dropout))
end

# transformers' blocks use `nn.LayerNorm`'s default eps, not their config's `layer_norm_eps`
sam3_norm(d) = LayerNorm(d; eps = 1.0f-5)

function decoder_mlp(dims, init)
    layers = [Dense(dims[i] => dims[i + 1]; init) for i in 1:(length(dims) - 1)]
    return Sam3DecoderMLP(layers[1], layers[2], length(layers) == 3 ? layers[3] : nothing)
end

function sam3_geometry_encoder(config)
    c = sam3_block_config(config,
        merge(SAM3_ENCODER_DEFAULTS, (; num_layers = 3, roi_size = 7)))
    init = normal_init(c.initializer_range)
    d = c.hidden_size
    layers = [Sam3GeometryEncoderLayer(sam3_norm(d), sam3_attention(c, init),
                  dropout_or_identity(c.dropout), sam3_attention(c, init), sam3_norm(d),
                  sam3_mlp(c, init), sam3_norm(d))
              for _ in 1:(c.num_layers)]
    return Sam3GeometryEncoder(Sam3SinePositionEmbedding(d ÷ 2, 10_000, true, 2π),
        Embedding(2 => d; init), Embedding(1 => d; init), Dense(4 => d; init),
        Conv((c.roi_size, c.roi_size), d => d), Dense(d + 2 => d; init), sam3_norm(d),
        Dense(d => d; init), sam3_norm(d), layers, sam3_norm(d), c.roi_size)
end

function sam3_detr_encoder(config)
    c = sam3_block_config(config, SAM3_ENCODER_DEFAULTS)
    init = normal_init(c.initializer_range)
    d = c.hidden_size
    return Sam3DetrEncoder([Sam3DetrEncoderLayer(sam3_norm(d), sam3_attention(c, init),
                                dropout_or_identity(c.dropout), sam3_attention(c, init),
                                sam3_norm(d), sam3_mlp(c, init), sam3_norm(d))
                            for _ in 1:(c.num_layers)])
end

function sam3_detr_decoder(config)
    c = sam3_block_config(config, merge(SAM3_ENCODER_DEFAULTS, (; num_queries = 200)))
    init = normal_init(c.initializer_range)
    d, heads = c.hidden_size, c.num_attention_heads
    drop() = dropout_or_identity(c.dropout)
    layers = [Sam3DetrDecoderLayer(sam3_attention(c, init), drop(), sam3_norm(d),
                  sam3_attention(c, init), drop(), sam3_norm(d), sam3_attention(c, init),
                  drop(), sam3_norm(d), sam3_mlp(c, init), sam3_norm(d), drop())
              for _ in 1:(c.num_layers)]
    return Sam3DetrDecoder(layers, sam3_norm(d), decoder_mlp((d, d, d, 4), init),
        Embedding(c.num_queries => d; init), Embedding(c.num_queries => 4; init),
        Embedding(1 => d; init), decoder_mlp((d, d, d, 1), init), sam3_norm(d),
        decoder_mlp((2d, d, d), init), decoder_mlp((2, d, heads), init),
        decoder_mlp((2, d, heads), init),
        Sam3SinePositionEmbedding(d ÷ 2, 10_000, false, 2π), 10.0)
end

function sam3_mask_decoder(config)
    c = sam3_block_config(config, (; hidden_size = 256, num_upsampling_stages = 3,
        dropout = 0.0, num_attention_heads = 8, initializer_range = 0.02))
    init = normal_init(c.initializer_range)
    d = c.hidden_size
    d % 8 == 0 || error("the mask decoder's `hidden_size = $d` must be divisible by 8, its groups of channels")
    pixel_decoder = Sam3PixelDecoder(
        [Conv((3, 3), d => d; pad = 1) for _ in 1:(c.num_upsampling_stages)],
        [GroupNorm(d, 8) for _ in 1:(c.num_upsampling_stages)])
    return Sam3MaskDecoder(pixel_decoder,
        Sam3MaskEmbedder([Dense(d => d; init) for _ in 1:3], relu), Conv((1, 1), d => d),
        Conv((1, 1), d => 1), sam3_attention(c, init), sam3_norm(d),
        dropout_or_identity(c.dropout))
end

function sam3_dot_product_scoring(config)
    c = sam3_block_config(config, merge(SAM3_ENCODER_DEFAULTS, (; num_queries = 200)))
    init = normal_init(c.initializer_range)
    d = c.hidden_size
    return Sam3DotProductScoring(decoder_mlp((d, c.intermediate_size, d), init),
        dropout_or_identity(c.dropout), sam3_norm(d), Dense(d => d; init),
        Dense(d => d; init), 1 / sqrt(d), 12.0)
end

# The text encoder of a `Sam3Config` without a `text_config`
const SAM3_TEXT_DEFAULTS = (; vocab_size = 49408, hidden_size = 1024,
    intermediate_size = 4096, projection_dim = 512, num_hidden_layers = 24,
    num_attention_heads = 16, max_position_embeddings = 32, hidden_act = "gelu")

"""
    sam3_model(; vision_config = nothing, text_config = nothing,
               geometry_encoder_config = nothing, detr_encoder_config = nothing,
               detr_decoder_config = nothing, mask_decoder_config = nothing)

Port of transformers' `Sam3Model`, the detector of SAM 3 (Carion et al., 2025), built from
its `Sam3Config`: NamedTuples of the fields of the configs of its parts (other keys are
ignored), each `nothing` for the defaults of transformers. The vision encoder
[`sam3_vision_model`](@ref Metalhead.Transformers.sam3_vision_model) of `vision_config`
and the CLIP text encoder [`clip_text_model_with_projection`](@ref
Metalhead.Transformers.clip_text_model_with_projection) of `text_config` feed a DETR
encoder and decoder, whose queries the mask decoder turns into masks; box prompts go
through a geometry encoder.

The model maps images and text prompts, and optionally box prompts, to the masks, boxes and
scores of the objects they describe: see [`Sam3Model`](@ref
Metalhead.Transformers.Sam3Model) for its inputs and outputs,
[`sam3_image_inputs`](@ref Metalhead.Transformers.sam3_image_inputs) to make its images,
and [`post_process_instance_segmentation`](@ref
Metalhead.Transformers.post_process_instance_segmentation) to read its outputs.
"""
function sam3_model(; vision_config = nothing, text_config = nothing,
        geometry_encoder_config = nothing, detr_encoder_config = nothing,
        detr_decoder_config = nothing, mask_decoder_config = nothing,
        initializer_range = 0.02)
    vision_encoder = sam3_vision_model(; config_args(sam3_vision_model, vision_config)...)
    text = text_config === nothing ? SAM3_TEXT_DEFAULTS : text_config
    text_encoder = clip_text_model_with_projection(;
        config_args(clip_text_model_with_projection, text)...)
    text_hidden = size(text_encoder.text_projection.weight, 2)
    d = sam3_block_config(detr_encoder_config, SAM3_ENCODER_DEFAULTS).hidden_size
    text_projection = Dense(text_hidden => d; init = normal_init(initializer_range))
    return Sam3Model(vision_encoder, text_encoder, text_projection,
        sam3_geometry_encoder(geometry_encoder_config),
        sam3_detr_encoder(detr_encoder_config),
        sam3_detr_decoder(detr_decoder_config), sam3_mask_decoder(mask_decoder_config),
        sam3_dot_product_scoring(detr_decoder_config))
end

register!(ArchSpec(TRANSFORMERS, "Sam3Model", sam3_model; family = "sam3",
    references = ["unsloth/sam3.1" => "cd3538905f98152ba184d2791fc5aa5d06e2ae45"]))

"""
    Sam3VideoModel

transformers' `Sam3VideoModel`, the model of the checkpoints of SAM 3 (`jetjodh/sam3`),
which detects objects in each frame with its `detector_model`, a [`Sam3Model`](@ref
Metalhead.Transformers.Sam3Model), and tracks them across frames. Only the detector is
ported: the model is called and post-processed as its detector, on images.
"""
@concrete struct Sam3VideoModel
    detector_model
end
Flux.@layer Sam3VideoModel

(m::Sam3VideoModel)(args...; kwargs...) = m.detector_model(args...; kwargs...)
function get_text_features(m::Sam3VideoModel, input_ids; attention_mask = nothing)
    return get_text_features(m.detector_model, input_ids; attention_mask)
end
get_vision_features(m::Sam3VideoModel, pixel_values) = get_vision_features(
    m.detector_model, pixel_values)

"""
    sam3_video_model(; detector_config = nothing, tracker_config = nothing)

Port of the detector of transformers' `Sam3VideoModel` (see [`Sam3VideoModel`](@ref
Metalhead.Transformers.Sam3VideoModel)), built from its `Sam3VideoConfig`: the
[`sam3_model`](@ref Metalhead.Transformers.sam3_model) of `detector_config`, a NamedTuple of
the fields of its `Sam3Config`. The tracker of `tracker_config` is not ported, and the
weights of `tracker_model` and `tracker_neck` are not loaded.
"""
function sam3_video_model(; detector_config = nothing, tracker_config = nothing)
    return Sam3VideoModel(sam3_model(; config_args(sam3_model, detector_config)...))
end

# jetjodh/sam3 is an ungated copy of the gated facebook/sam3, with the same weights
register!(ArchSpec(TRANSFORMERS, "Sam3VideoModel", sam3_video_model; family = "sam3",
    ignore = [r"^tracker_model\.", r"^tracker_neck\."],
    references = ["jetjodh/sam3" => "1aa50ce07302cb375f85d8084b68a0fb378b8d85"]))
