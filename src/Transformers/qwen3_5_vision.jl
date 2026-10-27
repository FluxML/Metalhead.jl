# Ported from transformers.models.qwen3_5.modeling_qwen3_5 (transformers 5.20, Apache-2.0,
# The Hugging Face Team): the vision encoder of Qwen3.5, `Qwen3_5VisionModel`. Validated
# against test/fixtures/qwen3_5_vision_*.

# torch's `view(-1, in_channels, temporal_patch_size, patch_size, patch_size)` of the flat
# patches, then a Conv3d whose kernel is a whole patch.
@concrete struct Qwen3_5VisionPatchEmbed
    proj
end
Flux.@layer Qwen3_5VisionPatchEmbed

function (m::Qwen3_5VisionPatchEmbed)(x)
    kw, kh, kt, c, _ = size(m.proj.weight)
    T = eltype(m.proj.weight)
    x = eltype(x) == T ? x : T.(x)
    return reshape(m.proj(reshape(x, kw, kh, kt, c, :)), :, size(x, 2))
end

@concrete struct Qwen3_5VisionMLP
    linear_fc1
    linear_fc2
    act_fn
end
Flux.@layer Qwen3_5VisionMLP

(m::Qwen3_5VisionMLP)(x) = m.linear_fc2(m.act_fn(m.linear_fc1(x)))

# Attention within each frame of each image only: `seqlens` are the lengths of the frames,
# whose patches follow one another (transformers' `cu_seqlens`).
@concrete struct Qwen3_5VisionAttention
    num_heads
    qkv
    proj
end
Flux.@layer Qwen3_5VisionAttention

function (m::Qwen3_5VisionAttention)(x, position_embeddings, seqlens)
    n = size(x, 2)
    cos, sin = position_embeddings
    qkv = reshape(m.qkv(x), :, m.num_heads, 3, n)
    # in the eltype of the tables, at least Float32, as transformers rotates
    rotate(a) = convert_like(qkv, NNlib.apply_rotary_embedding(a, cos, sin))
    q, k = rotate(qkv[:, :, 1, :]), rotate(qkv[:, :, 2, :])
    y = NNlib.scaled_dot_product_attention(q, k, qkv[:, :, 3, :]; q_segments = seqlens)
    return m.proj(reshape(y, :, n))
end

convert_like(x, y) = eltype(y) == eltype(x) ? y : eltype(x).(y)

@concrete struct Qwen3_5VisionBlock
    norm1
    norm2
    attn
    mlp
end
Flux.@layer Qwen3_5VisionBlock

function (m::Qwen3_5VisionBlock)(x, position_embeddings, seqlens)
    x = x .+ m.attn(m.norm1(x), position_embeddings, seqlens)
    return x .+ m.mlp(m.norm2(x))
end

# The LayerNorm is applied to each patch, before the patches of each merge window are
# concatenated (transformers' `use_postshuffle_norm = False`).
@concrete struct Qwen3_5VisionPatchMerger
    norm
    linear_fc1
    act_fn
    linear_fc2
end
Flux.@layer Qwen3_5VisionPatchMerger

function (m::Qwen3_5VisionPatchMerger)(x)
    x = reshape(m.norm(x), size(m.linear_fc1.weight, 2), :)
    return m.linear_fc2(m.act_fn(m.linear_fc1(x)))
end

# The 2D axial rotary embedding: the first half of the rotated pairs of features turn with
# the row of the patch, the second half with its column, at the same frequencies.
@concrete struct Qwen3_5VisionRotaryEmbedding
    head_dim
    rope_theta
end
Flux.@layer Qwen3_5VisionRotaryEmbedding

# The tables of `NNlib.apply_rotary_embedding` for the patches at the 0-based `rows` and
# `cols`, computed in Float64 (transformers computes them in Float32) and returned in at
# least Float32, on the device of `x`.
function axial_rotary_tables(m, rows, cols, x)
    return ChainRulesCore.ignore_derivatives() do
        spatial = m.head_dim ÷ 2
        inv_freq = Float64(m.rope_theta) .^ (-(0:2:(spatial - 2)) ./ spatial)
        angles = [inv_freq .* rows'; inv_freq .* cols']
        T = promote_type(eltype(x), Float32)
        table(a) = copyto!(similar(x, T, size(a)), T.(a))
        return table(cos.(angles)), table(sin.(angles))
    end
end

@concrete struct Qwen3_5VisionModel
    patch_embed
    pos_embed
    rotary_pos_emb
    blocks
    merger
    spatial_merge_size
end
Flux.@layer Qwen3_5VisionModel

# The (t, h, w) grids of the columns of `grid_thw`, checked against the number of patches.
function vision_grids(grid_thw, num_patches, merge)
    size(grid_thw, 1) == 3 ||
        error("`grid_thw` must have size (3, num_images), got $(size(grid_thw))")
    grids = [Tuple(Int.(g)) for g in eachcol(Array(grid_thw))]
    total = sum(prod, grids; init = 0)
    total == num_patches ||
        error("`grid_thw` has $total patches in all, `pixel_values` $num_patches")
    for (t, h, w) in grids
        t > 0 && h % merge == 0 && w % merge == 0 ||
            error("the grid (t, h, w) = $((t, h, w)) of `grid_thw` must have t > 0 and h and w divisible by `spatial_merge_size` = $merge")
    end
    return grids
end
ChainRulesCore.@non_differentiable vision_grids(::Any...)

# The 0-based rows and columns of the patches of each frame in the order of the image
# processor: merge window by merge window (transformers' `get_vision_position_ids`).
function patch_positions(grids, merge)
    order = [(br * merge + ir, bc * merge + ic)
             for (t, h, w) in grids for _ in 1:t
             for br in 0:(h ÷ merge - 1) for bc in 0:(w ÷ merge - 1)
             for ir in 0:(merge - 1) for ic in 0:(merge - 1)]
    return first.(order), last.(order)
end
ChainRulesCore.@non_differentiable patch_positions(::Any...)

# transformers' `get_vision_interpolation_indices_and_weights` for bilinear interpolation
# with aligned corners: the 4 entries of the (side, side) table of learned position
# embeddings around each patch, as 1-based indices into the table, and their weights, in
# Float64 (transformers computes them in Float32). Both have size (4, num_patches).
function interpolation_taps(grids, rows, cols, side)
    heights = [h for (t, h, w) in grids for _ in 1:(t * h * w)]
    widths = [w for (t, h, w) in grids for _ in 1:(t * h * w)]
    function axis(i, n)
        src = i * (side - 1) / max(n - 1, 1)
        f = floor(src)
        lo = clamp(Int(f), 0, side - 1)
        return (lo, min(lo + 1, side - 1)), (1 - (src - f), src - f)
    end
    index = Matrix{Int}(undef, 4, length(rows))
    weight = Matrix{Float64}(undef, 4, length(rows))
    for j in eachindex(rows)
        (h0, h1), (a0, a1) = axis(rows[j], heights[j])
        (w0, w1), (b0, b1) = axis(cols[j], widths[j])
        index[:, j] .= (h0 * side + w0, h0 * side + w1, h1 * side + w0, h1 * side + w1) .+ 1
        weight[:, j] .= (a0 * b0, a0 * b1, a1 * b0, a1 * b1)
    end
    return index, weight
end
ChainRulesCore.@non_differentiable interpolation_taps(::Any...)

# The learned position embeddings resampled to the grid of each image, weighted in at least
# Float32, as transformers' Float32 weights do, and returned in the eltype of `x`.
function interpolated_pos_embed(pos_embed, grids, rows, cols, x)
    side = isqrt(size(pos_embed.weight, 2))
    index, weight = ChainRulesCore.ignore_derivatives() do
        index, weight = interpolation_taps(grids, rows, cols, side)
        W = promote_type(eltype(x), Float32)
        return device_indices(x, index), copyto!(similar(x, W, size(weight)), weight)
    end
    e = pos_embed(index)
    return convert_like(x, dropdims(
        sum(e .* reshape(weight, 1, size(weight)...); dims = 2);
        dims = 2))
end

"""
    (model::Qwen3_5VisionModel)(pixel_values, grid_thw) -> (; last_hidden_state, pooler_output)

Encode images and videos, given as patches: `pixel_values` of size
`(in_channels * temporal_patch_size * patch_size^2, num_patches)`, and `grid_thw`, an
integer matrix of size `(3, num_images)` whose column `(t, h, w)` gives the numbers of
patches of an image along time, height and width (see [`qwen3_5_vision_model`](@ref
Metalhead.Transformers.qwen3_5_vision_model)). Returns the hidden states of the patches,
of size `(hidden_size, num_patches)`, and the merged tokens of the `pooler_output`, of size
`(out_hidden_size, num_patches ÷ spatial_merge_size^2)`, image after image.
"""
function (m::Qwen3_5VisionModel)(pixel_values::AbstractMatrix,
        grid_thw::AbstractMatrix{<:Integer})
    merge = m.spatial_merge_size
    grids = vision_grids(grid_thw, size(pixel_values, 2), merge)
    rows, cols = patch_positions(grids, merge)
    seqlens = ChainRulesCore.ignore_derivatives(() -> [h * w for (t, h, w) in grids
                                                       for _ in 1:t])
    x = m.patch_embed(pixel_values)
    x = x .+ interpolated_pos_embed(m.pos_embed, grids, rows, cols, x)
    position_embeddings = axial_rotary_tables(m.rotary_pos_emb, rows, cols, x)
    for block in m.blocks
        x = block(x, position_embeddings, seqlens)
    end
    return (; last_hidden_state = x, pooler_output = m.merger(x))
end

"""
    (model::Qwen3_5VisionModel)(inputs::NamedTuple)

The model on the output of a [`Qwen2VLImageProcessor`](@ref
Metalhead.Transformers.Qwen2VLImageProcessor) (or of a [`Qwen3VLProcessor`](@ref
Metalhead.Transformers.Qwen3VLProcessor) with images): its `pixel_values` and
`image_grid_thw`; other keys are ignored.
"""
function (m::Qwen3_5VisionModel)(inputs::NamedTuple)
    check_inputs(m, inputs, (:pixel_values, :image_grid_thw);
        hint = "pass `images` to the processor")
    return m(inputs.pixel_values, inputs.image_grid_thw)
end

# The `rope_theta` of the axial rotary embedding, from `rope_parameters` or the older
# `rope_theta` and `rope_scaling` keys of config.json; the config makes a "default"
# `rope_type` "axial", the only one transformers supports.
function axial_rope_theta(; rope_theta, rope_scaling, rope_parameters)
    params = something(rope_scaling, rope_parameters, (;))
    params = (; (Symbol(k) => v for (k, v) in pairs(params))...)
    rope_type = get(params, :rope_type, get(params, :type, "default"))
    rope_type in ("default", "axial") ||
        error("`rope_type = $(repr(rope_type))` is not supported by Qwen3_5VisionModel; supported: \"axial\"")
    return Float64(get(params, :rope_theta, something(rope_theta, 10_000.0)))
end

"""
    qwen3_5_vision_model(; depth = 27, hidden_size = 1152, num_heads = 16, patch_size = 16,
                         spatial_merge_size = 2, temporal_patch_size = 2, kwargs...)

Port of transformers' `Qwen3_5VisionModel`, the vision encoder of Qwen3.5, built from the
fields of its `Qwen3_5VisionConfig` (the `vision_config` of a Qwen3.5 checkpoint) with the
same names and defaults. `deepstack_visual_indexes`, which config.json may carry, must be
empty, as transformers' Qwen3.5 has no deepstack mergers; the rotary embedding is the axial
one (`rope_type = "axial"`, or `"default"`).

The model takes `(pixel_values, grid_thw)`, as the image processor of a checkpoint
([`Qwen2VLImageProcessor`](@ref Metalhead.Transformers.Qwen2VLImageProcessor)) makes them,
whose output it also takes as it is (`model(ip(images))`), and returns the NamedTuple
`(; last_hidden_state, pooler_output)`:

  - `pixel_values`, of size
    `(in_channels * temporal_patch_size * patch_size^2, num_patches)`, is the transpose
    of transformers' `pixel_values`: each column is a patch, flattened as
    torch's `(in_channels, temporal_patch_size, patch_size, patch_size)` (that is, as a
    Julia array of size `(patch_size, patch_size, temporal_patch_size, in_channels)`, width
    first).
  - `grid_thw`, an integer matrix of size `(3, num_images)` (transformers' `grid_thw`
    transposed), has a column `(t, h, w)` for each image or video: its number of patches
    along time (frames of `temporal_patch_size`), height and width. The patches of the
    images follow one another; within an image, frame after frame, each frame is ordered by
    windows of `spatial_merge_size × spatial_merge_size` patches, row-major over the windows
    and row-major within each window.
  - `last_hidden_state` has size `(hidden_size, num_patches)`. `pooler_output`, the merged
    tokens that the language model takes, image after image, has size
    `(out_hidden_size, num_patches ÷ spatial_merge_size^2)`.

Each frame of each image attends to its own patches only.
"""
function qwen3_5_vision_model(; depth = 27, hidden_size = 1152,
        hidden_act = "gelu_pytorch_tanh", intermediate_size = 4304, num_heads = 16,
        num_attention_heads = nothing, in_channels = 3, patch_size = 16,
        spatial_merge_size = 2, temporal_patch_size = 2, out_hidden_size = 3584,
        num_position_embeddings = 2304, initializer_range = 0.02, rope_parameters = nothing,
        rope_theta = nothing, rope_scaling = nothing, head_dim = nothing,
        deepstack_visual_indexes = ())
    isempty(deepstack_visual_indexes) ||
        error("`deepstack_visual_indexes = $(repr(deepstack_visual_indexes))` is not supported: transformers' Qwen3_5VisionModel has no deepstack mergers")
    patch_size isa Integer && temporal_patch_size isa Integer ||
        error("`patch_size` and `temporal_patch_size` must be integers")
    num_heads = something(num_attention_heads, num_heads)
    hidden_size % num_heads == 0 ||
        error("`hidden_size` ($hidden_size) must be divisible by `num_heads` ($num_heads)")
    dim = hidden_size ÷ num_heads
    something(head_dim, dim) == dim ||
        error("`head_dim` ($head_dim) must be `hidden_size ÷ num_heads` ($dim)")
    dim % 4 == 0 ||
        error("the head dimension `hidden_size ÷ num_heads` ($dim) must be divisible by 4")
    theta = axial_rope_theta(; rope_theta, rope_scaling, rope_parameters)
    act = act_fn(hidden_act)
    init = normal_init(initializer_range)
    dense(in_out) = init_layer(Dense, in_out; init)
    norm(d) = init_layer(LayerNorm, d; eps = 1e-6)
    kernel = (patch_size, patch_size, temporal_patch_size)
    patch_embed = Qwen3_5VisionPatchEmbed(init_layer(
        Conv, kernel, in_channels => hidden_size;
        stride = kernel, init))
    block() = Qwen3_5VisionBlock(norm(hidden_size), norm(hidden_size),
        Qwen3_5VisionAttention(num_heads, dense(hidden_size => 3hidden_size),
            dense(hidden_size => hidden_size)),
        Qwen3_5VisionMLP(dense(hidden_size => intermediate_size),
            dense(intermediate_size => hidden_size), act))
    merged = hidden_size * spatial_merge_size^2
    merger = Qwen3_5VisionPatchMerger(norm(hidden_size), dense(merged => merged), gelu_erf,
        dense(merged => out_hidden_size))
    return Qwen3_5VisionModel(patch_embed,
        init_layer(Embedding, num_position_embeddings => hidden_size; init),
        Qwen3_5VisionRotaryEmbedding(dim, theta), [block() for _ in 1:depth], merger,
        spatial_merge_size)
end

register!(ArchSpec(TRANSFORMERS, "Qwen3_5VisionModel", qwen3_5_vision_model;
    family = "qwen"))
