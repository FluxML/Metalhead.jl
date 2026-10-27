# Ported from transformers.models.clip.modeling_clip (transformers 5.20, Apache-2.0, The
# Hugging Face Team): `CLIPModel`, the dual encoder of CLIP (Radford et al., 2021), its text
# and vision encoders alone (`CLIPTextModel`, `CLIPVisionModel`) or with their projections
# (`CLIPTextModelWithProjection`, the text encoder of SAM 3, and
# `CLIPVisionModelWithProjection`). Not ported: `CLIPForImageClassification`, the
# `position_ids` of the text model and the contrastive loss. Validated against
# test/fixtures/clip_* and sam3_*, and the references of the registered checkpoints.

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

@concrete struct CLIPVisionEmbeddings
    class_embedding
    patch_embedding
    position_embedding
    image_size
    patch_size
end
Flux.@layer CLIPVisionEmbeddings

# transformers' `interpolate_pos_encoding`: the position embeddings of the square grid of
# the checkpoint, resized bicubically to the grid of an image of `height × width` pixels.
# The embeddings of the checkpoint stay as they are for its number of patches in a square.
function position_encoding(m::CLIPVisionEmbeddings, num_patches, height, width,
        interpolate_pos_encoding)
    weight = m.position_embedding.weight
    d, num_positions = size(weight, 1), size(weight, 2) - 1
    (!interpolate_pos_encoding || num_patches == num_positions && height == width) &&
        return weight
    side = isqrt(num_positions)
    grid = permutedims(reshape(weight[:, 2:end], d, side, side), (2, 3, 1))
    new_w, new_h = width ÷ m.patch_size, height ÷ m.patch_size
    resized = NNlib.upsample_bicubic(reshape(grid, side, side, d, 1);
        size = (new_w, new_h), align_corners = false)
    patch_pos_embed = permutedims(reshape(resized, new_w, new_h, d), (3, 1, 2))
    return hcat(weight[:, 1:1], reshape(patch_pos_embed, d, :))
end

function (m::CLIPVisionEmbeddings)(pixel_values; interpolate_pos_encoding::Bool = false)
    width, height = size(pixel_values, 1), size(pixel_values, 2)
    s = m.image_size
    interpolate_pos_encoding || (height, width) == (s, s) ||
        error("the images are $height × $width pixels (height × width), but the model takes $s × $s: resize them, or pass `interpolate_pos_encoding = true`")
    T = eltype(m.patch_embedding.weight)
    x = eltype(pixel_values) === T ? pixel_values : T.(pixel_values)
    patches = m.patch_embedding(x)
    gw, gh, d, b = size(patches)
    patch_embeds = permutedims(reshape(patches, gw * gh, d, b), (2, 1, 3))
    class_embeds = repeat(reshape(m.class_embedding, d, 1, 1), 1, 1, b)
    embeddings = cat(class_embeds, patch_embeds; dims = 2)
    return embeddings .+
           position_encoding(m, gw * gh, height, width, interpolate_pos_encoding)
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

# Self-attention over the first `kv_lengths` tokens of each sequence (all without), causal
# in the text encoder.
function (m::CLIPAttention)(x, kv_lengths = nothing; is_causal::Bool = false)
    _, n, b = size(x)
    heads(p) = reshape(p(x), :, m.num_heads, n, b)
    q, k, v = heads(m.q_proj), heads(m.k_proj), heads(m.v_proj)
    y = NNlib.scaled_dot_product_attention(q, k, v; is_causal, kv_lengths,
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

function (m::CLIPEncoderLayer)(x, kv_lengths = nothing; is_causal::Bool = false)
    x = x .+ m.self_attn(m.layer_norm1(x), kv_lengths; is_causal)
    return x .+ m.mlp(m.layer_norm2(x))
end

@concrete struct CLIPEncoder
    layers
end
Flux.@layer CLIPEncoder

function (m::CLIPEncoder)(x, kv_lengths = nothing; is_causal::Bool = false)
    for layer in m.layers
        x = layer(x, kv_lengths; is_causal)
    end
    return x
end

"""
    CLIPTextModel

transformers' `CLIPTextModel`, the text encoder of CLIP, built by
[`clip_text_model`](@ref Metalhead.Transformers.clip_text_model).

    (model::CLIPTextModel)(input_ids; attention_mask = nothing)

The `last_hidden_state` (hidden_size, seq_len, batch_size), after the final norm, and the
`pooler_output` (hidden_size, batch_size), its features at the end-of-text token of each
sequence, of the token indices `input_ids` (seq_len, batch_size), 1-based, padded on the
right after the ones of `attention_mask`. The attention is causal.
"""
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

function (m::CLIPTextModel)(input_ids; attention_mask = nothing)
    kv_lengths = kv_lengths_of(attention_mask, input_ids)
    x = m.encoder(m.embeddings(input_ids), kv_lengths; is_causal = true)
    x = m.final_layer_norm(x)
    d, n, b = size(x)
    index = ChainRulesCore.ignore_derivatives() do
        return device_indices(x, eos_positions(input_ids, m.eos_token_id) .+
                                 n .* (0:(b - 1)))
    end
    return (; last_hidden_state = x, pooler_output = NNlib.gather(reshape(x, d, :), index))
end

"""
    CLIPVisionModel

transformers' `CLIPVisionModel`, the vision encoder of CLIP, a ViT, built by
[`clip_vision_model`](@ref Metalhead.Transformers.clip_vision_model).

    (model::CLIPVisionModel)(pixel_values; interpolate_pos_encoding = false)

The `last_hidden_state` (hidden_size, 1 + num_patches, batch_size), the class token then
the patches, row by row, and the `pooler_output` (hidden_size, batch_size), the class token
after the final norm, of the images `pixel_values` (width, height, channels, batch_size),
as the model's processor makes them. The images must have the `image_size` of the model,
unless `interpolate_pos_encoding = true`, which resizes the position embeddings to the grid
of their patches, bicubically.
"""
@concrete struct CLIPVisionModel
    embeddings
    pre_layrnorm
    encoder
    post_layernorm
end
Flux.@layer CLIPVisionModel

function (m::CLIPVisionModel)(pixel_values; interpolate_pos_encoding::Bool = false)
    x = m.pre_layrnorm(m.embeddings(pixel_values; interpolate_pos_encoding))
    last_hidden_state = m.encoder(x)
    return (; last_hidden_state,
        pooler_output = m.post_layernorm(last_hidden_state[:, 1, :]))
end

# Checkpoints of `CLIPTextModel` and `CLIPVisionModel` saved before transformers 5 hold the
# weights under `text_model.` and `vision_model.`, which transformers removes on loading
# (`PrefixChange` in conversion_mapping.py).
function checkpoint_prefix(::CLIPTextModel, ckpt)
    return any(startswith("text_model."), keys(ckpt)) ? "text_model" : ""
end
function checkpoint_prefix(::CLIPVisionModel, ckpt)
    return any(startswith("vision_model."), keys(ckpt)) ? "vision_model" : ""
end

"""
    CLIPTextModelWithProjection

transformers' `CLIPTextModelWithProjection`, the text encoder of CLIP with its projection,
built by [`clip_text_model_with_projection`](@ref
Metalhead.Transformers.clip_text_model_with_projection).

    (model::CLIPTextModelWithProjection)(input_ids; attention_mask = nothing)

The `text_embeds` (projection_dim, batch_size), the projected `pooler_output`, and the
`last_hidden_state` of its [`CLIPTextModel`](@ref Metalhead.Transformers.CLIPTextModel).
"""
@concrete struct CLIPTextModelWithProjection
    text_model
    text_projection
end
Flux.@layer CLIPTextModelWithProjection

function (m::CLIPTextModelWithProjection)(input_ids; attention_mask = nothing)
    y = m.text_model(input_ids; attention_mask)
    return (; text_embeds = m.text_projection(y.pooler_output),
        last_hidden_state = y.last_hidden_state)
end

"""
    CLIPVisionModelWithProjection

transformers' `CLIPVisionModelWithProjection`, the vision encoder of CLIP with its
projection, built by [`clip_vision_model_with_projection`](@ref
Metalhead.Transformers.clip_vision_model_with_projection).

    (model::CLIPVisionModelWithProjection)(pixel_values; interpolate_pos_encoding = false)

The `image_embeds` (projection_dim, batch_size), the projected `pooler_output`, and the
`last_hidden_state` of its [`CLIPVisionModel`](@ref Metalhead.Transformers.CLIPVisionModel).
"""
@concrete struct CLIPVisionModelWithProjection
    vision_model
    visual_projection
end
Flux.@layer CLIPVisionModelWithProjection

function (m::CLIPVisionModelWithProjection)(pixel_values;
        interpolate_pos_encoding::Bool = false)
    y = m.vision_model(pixel_values; interpolate_pos_encoding)
    return (; image_embeds = m.visual_projection(y.pooler_output),
        last_hidden_state = y.last_hidden_state)
end

"""
    CLIPModel

transformers' `CLIPModel`, the dual encoder of CLIP, built by
[`clip_model`](@ref Metalhead.Transformers.clip_model): a [`CLIPTextModel`](@ref
Metalhead.Transformers.CLIPTextModel) and a [`CLIPVisionModel`](@ref
Metalhead.Transformers.CLIPVisionModel), whose pooled features `text_projection` and
`visual_projection` project to a space they share, and the `logit_scale` of their
similarities.

    (model::CLIPModel)(input_ids, pixel_values; attention_mask = nothing,
                       interpolate_pos_encoding = false)
    (model::CLIPModel)(inputs::NamedTuple)

The similarities of the texts `input_ids` (seq_len, num_texts), token indices, 1-based,
padded on the right after the ones of `attention_mask`, and of the images `pixel_values`
(width, height, channels, num_images), as the model's processor, a [`CLIPProcessor`](@ref
Metalhead.Transformers.CLIPProcessor), makes them; or of the output of the processor,
`model(proc(; text, images))`. Returns a NamedTuple with the keys of transformers'
`CLIPOutput`, in Julia's layout (PyTorch's dimensions reversed):

| Key                | Size                           | Contents                                                                                                                                         |
|:------------------ |:------------------------------ |:------------------------------------------------------------------------------------------------------------------------------------------------ |
| `logits_per_image` | `(num_texts, num_images)`      | the cosine similarity of each text and image, times `exp(logit_scale)`; a softmax over its first dimension classifies each image among the texts |
| `logits_per_text`  | `(num_images, num_texts)`      | its transpose                                                                                                                                    |
| `text_embeds`      | `(projection_dim, num_texts)`  | the projected features of the texts, of unit norm                                                                                                |
| `image_embeds`     | `(projection_dim, num_images)` | the projected features of the images, of unit norm                                                                                               |

The outputs of the encoders, unnormalised, are those of [`get_text_features`](@ref
Metalhead.Transformers.get_text_features) and [`get_image_features`](@ref
Metalhead.Transformers.get_image_features).
"""
@concrete struct CLIPModel
    text_model
    vision_model
    visual_projection
    text_projection
    logit_scale
end
Flux.@layer CLIPModel

"""
    get_text_features(model::CLIPModel, input_ids; attention_mask = nothing)
    get_text_features(model::CLIPModel, inputs::NamedTuple)

The output of the text encoder of CLIP on the token indices `input_ids` (seq_len,
num_texts), 1-based and padded on the right after the ones of `attention_mask`, or on the
`input_ids` (and `attention_mask`) of the output of its processor: its `last_hidden_state`
and, as `pooler_output` (projection_dim, num_texts), its pooled features projected by
`text_projection`, not normalised (transformers' `get_text_features`).
"""
function get_text_features(m::CLIPModel, input_ids; attention_mask = nothing)
    y = m.text_model(input_ids; attention_mask)
    return (; y.last_hidden_state, pooler_output = m.text_projection(y.pooler_output))
end

function get_text_features(m::CLIPModel, inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids,); hint = "pass `text` to the processor")
    return get_text_features(m, inputs.input_ids;
        attention_mask = get(inputs, :attention_mask, nothing))
end

"""
    get_image_features(model::CLIPModel, pixel_values; interpolate_pos_encoding = false)
    get_image_features(model::CLIPModel, inputs::NamedTuple;
                       interpolate_pos_encoding = false)

The output of the vision encoder of CLIP on the images `pixel_values` (width, height,
channels, num_images), or on the `pixel_values` of the output of its processor: its
`last_hidden_state` and, as `pooler_output` (projection_dim, num_images), its pooled
features projected by `visual_projection`, not normalised (transformers'
`get_image_features`). See [`CLIPVisionModel`](@ref Metalhead.Transformers.CLIPVisionModel)
for `interpolate_pos_encoding`.
"""
function get_image_features(m::CLIPModel, pixel_values;
        interpolate_pos_encoding::Bool = false)
    y = m.vision_model(pixel_values; interpolate_pos_encoding)
    return (; y.last_hidden_state, pooler_output = m.visual_projection(y.pooler_output))
end

function get_image_features(m::CLIPModel, inputs::NamedTuple;
        interpolate_pos_encoding::Bool = false)
    check_inputs(m, inputs, (:pixel_values,); hint = "pass `images` to the processor")
    return get_image_features(m, inputs.pixel_values; interpolate_pos_encoding)
end

function (m::CLIPModel)(input_ids, pixel_values; attention_mask = nothing,
        interpolate_pos_encoding::Bool = false)
    image_embeds = get_image_features(m, pixel_values;
        interpolate_pos_encoding).pooler_output
    text_embeds = get_text_features(m, input_ids; attention_mask).pooler_output
    vector_norm(x) = sqrt.(sum(abs2, x; dims = 1))  # transformers' `_get_vector_norm`
    image_embeds = image_embeds ./ vector_norm(image_embeds)
    text_embeds = text_embeds ./ vector_norm(text_embeds)
    # `sum` of the 0-dimensional `logit_scale`, so that its gradient is an array too
    logits_per_text = (transpose(image_embeds) * text_embeds) .* exp(sum(m.logit_scale))
    return (; logits_per_image = permutedims(logits_per_text), logits_per_text,
        text_embeds, image_embeds)
end

const CLIP_HINT = "pass `text` and `images` to the processor"

function (m::CLIPModel)(inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids, :pixel_values); hint = CLIP_HINT)
    return m(inputs.input_ids, inputs.pixel_values;
        attention_mask = get(inputs, :attention_mask, nothing))
end
for T in (:CLIPTextModel, :CLIPTextModelWithProjection)
    @eval function (m::$T)(inputs::NamedTuple)
        check_inputs(m, inputs, (:input_ids,); hint = "pass `text` to the processor")
        return m(inputs.input_ids; attention_mask = get(inputs, :attention_mask, nothing))
    end
end
for T in (:CLIPVisionModel, :CLIPVisionModelWithProjection)
    @eval function (m::$T)(inputs::NamedTuple)
        check_inputs(m, inputs, (:pixel_values,); hint = "pass `images` to the processor")
        return m(inputs.pixel_values)
    end
end

# Builders

# The layers of the text or vision encoder, initialised as transformers' `_init_weights`.
function clip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout,
        initializer_factor)
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size = $hidden_size` must be divisible by `num_attention_heads = $num_attention_heads`")
    act = act_fn(hidden_act; table = SAM3_ACT2FN)
    in_std = hidden_size^-0.5 * (2 * num_hidden_layers)^-0.5 * initializer_factor
    out_std = hidden_size^-0.5 * initializer_factor
    fc_std = (2 * hidden_size)^-0.5 * initializer_factor
    dense(io, std) = init_layer(Dense, io; init = normal_init(std))
    norm() = init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps))
    layers = [CLIPEncoderLayer(
                  CLIPAttention(
                      num_attention_heads, dense(hidden_size => hidden_size, in_std),
                      dense(hidden_size => hidden_size, in_std),
                      dense(hidden_size => hidden_size, in_std),
                      dense(hidden_size => hidden_size, out_std),
                      dropout_or_identity(attention_dropout)),
                  norm(),
                  CLIPMLP(act, dense(hidden_size => intermediate_size, fc_std),
                      dense(intermediate_size => hidden_size, in_std)),
                  norm())
              for _ in 1:num_hidden_layers]
    return CLIPEncoder(layers)
end

function check_clip_size(name, v)
    v isa Integer && v > 0 ||
        error("`$name = $(repr(v))` is not supported: CLIP takes a square `$name`, a positive integer")
    return Int(v)
end

function clip_projection(io, factor)
    return init_layer(Dense, io; bias = false, init = normal_init(first(io)^-0.5 * factor))
end

"""
    clip_text_model(; vocab_size = 49408, hidden_size = 512, intermediate_size = 2048,
        projection_dim = 512, num_hidden_layers = 12, num_attention_heads = 8,
        max_position_embeddings = 77, hidden_act = "quick_gelu", layer_norm_eps = 1e-5,
        attention_dropout = 0.0, initializer_range = 0.02, initializer_factor = 1.0,
        eos_token_id = 49407)

Port of transformers' `CLIPTextModel`, the text encoder of CLIP, built from the fields of a
`CLIPTextConfig` (`projection_dim` is unused): a causal transformer over the token indices
`(seq_len, batch_size)` (1-based), pooled at the first end-of-text token `eos_token_id` of
each sequence, or at its largest token id with the `eos_token_id = 2` of older configs, as
in transformers (see [`CLIPTextModel`](@ref Metalhead.Transformers.CLIPTextModel)). The
activation `hidden_act` is `"quick_gelu"` in OpenAI's checkpoints and `"gelu"` (exact) in
others, such as LAION's.
"""
function clip_text_model(; vocab_size = 49408, hidden_size = 512, intermediate_size = 2048,
        projection_dim = 512, num_hidden_layers = 12, num_attention_heads = 8,
        max_position_embeddings = 77, hidden_act = "quick_gelu", layer_norm_eps = 1e-5,
        attention_dropout = 0.0, initializer_range = 0.02, initializer_factor = 1.0,
        eos_token_id = 49407)
    eos_token_id isa Integer ||
        error("`eos_token_id = $(repr(eos_token_id))` is not supported: CLIP pools at one end-of-text token, an integer")
    encoder = clip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout,
        initializer_factor)
    init = normal_init(0.02 * initializer_factor)
    embeddings = CLIPTextEmbeddings(init_layer(Embedding, vocab_size => hidden_size; init),
        init_layer(Embedding, max_position_embeddings => hidden_size; init))
    return CLIPTextModel(embeddings, encoder,
        init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps)), Int(eos_token_id))
end

"""
    clip_text_model_with_projection(; vocab_size = 49408, hidden_size = 512,
        intermediate_size = 2048, projection_dim = 512, num_hidden_layers = 12,
        num_attention_heads = 8, max_position_embeddings = 77, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, initializer_range = 0.02,
        initializer_factor = 1.0, eos_token_id = 49407)

Port of transformers' `CLIPTextModelWithProjection`, built from the fields of a
`CLIPTextConfig`: the text encoder [`clip_text_model`](@ref
Metalhead.Transformers.clip_text_model), whose pooled features are projected to
`projection_dim` (`text_embeds`, see [`CLIPTextModelWithProjection`](@ref
Metalhead.Transformers.CLIPTextModelWithProjection)).
"""
function clip_text_model_with_projection(; vocab_size = 49408, hidden_size = 512,
        intermediate_size = 2048, projection_dim = 512, num_hidden_layers = 12,
        num_attention_heads = 8, max_position_embeddings = 77, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, initializer_range = 0.02,
        initializer_factor = 1.0, eos_token_id = 49407)
    text_model = clip_text_model(; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads, max_position_embeddings, hidden_act,
        layer_norm_eps, attention_dropout, initializer_range, initializer_factor,
        eos_token_id)
    return CLIPTextModelWithProjection(text_model,
        clip_projection(hidden_size => projection_dim, initializer_factor))
end

"""
    clip_vision_model(; hidden_size = 768, intermediate_size = 3072, projection_dim = 512,
        num_hidden_layers = 12, num_attention_heads = 12, num_channels = 3,
        image_size = 224, patch_size = 32, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, initializer_range = 0.02,
        initializer_factor = 1.0)

Port of transformers' `CLIPVisionModel`, the vision encoder of CLIP, built from the fields
of a `CLIPVisionConfig` (`projection_dim` is unused): a ViT over the patches of
`patch_size × patch_size` pixels of square images of `image_size` pixels, with a class
token, a norm before the encoder (`pre_layrnorm`, as transformers spells it) and one of the
class token after it (see [`CLIPVisionModel`](@ref Metalhead.Transformers.CLIPVisionModel)).
"""
function clip_vision_model(; hidden_size = 768, intermediate_size = 3072,
        projection_dim = 512, num_hidden_layers = 12, num_attention_heads = 12,
        num_channels = 3, image_size = 224, patch_size = 32, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, initializer_range = 0.02,
        initializer_factor = 1.0)
    image_size = check_clip_size("image_size", image_size)
    p = check_clip_size("patch_size", patch_size)
    encoder = clip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout,
        initializer_factor)
    init = normal_init(initializer_range * initializer_factor)
    num_positions = (image_size ÷ p)^2 + 1
    embeddings = CLIPVisionEmbeddings(
        normal_init(hidden_size^-0.5 * initializer_factor)(hidden_size),
        init_layer(Conv, (p, p), num_channels => hidden_size; stride = p, bias = false,
            init),
        init_layer(Embedding, num_positions => hidden_size; init), image_size, p)
    norm() = init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps))
    return CLIPVisionModel(embeddings, norm(), encoder, norm())
end

"""
    clip_vision_model_with_projection(; hidden_size = 768, intermediate_size = 3072,
        projection_dim = 512, num_hidden_layers = 12, num_attention_heads = 12,
        num_channels = 3, image_size = 224, patch_size = 32, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, initializer_range = 0.02,
        initializer_factor = 1.0)

Port of transformers' `CLIPVisionModelWithProjection`, built from the fields of a
`CLIPVisionConfig`: the vision encoder [`clip_vision_model`](@ref
Metalhead.Transformers.clip_vision_model), whose pooled features are projected to
`projection_dim` (`image_embeds`, see [`CLIPVisionModelWithProjection`](@ref
Metalhead.Transformers.CLIPVisionModelWithProjection)).
"""
function clip_vision_model_with_projection(; hidden_size = 768, intermediate_size = 3072,
        projection_dim = 512, num_hidden_layers = 12, num_attention_heads = 12,
        num_channels = 3, image_size = 224, patch_size = 32, hidden_act = "quick_gelu",
        layer_norm_eps = 1e-5, attention_dropout = 0.0, initializer_range = 0.02,
        initializer_factor = 1.0)
    vision_model = clip_vision_model(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_channels, image_size, patch_size, hidden_act,
        layer_norm_eps, attention_dropout, initializer_range, initializer_factor)
    return CLIPVisionModelWithProjection(vision_model,
        clip_projection(hidden_size => projection_dim, initializer_factor))
end

"""
    clip_model(; text_config = nothing, vision_config = nothing, projection_dim = 512,
               logit_scale_init_value = 2.6592, initializer_factor = 1.0,
               text_config_dict = nothing, vision_config_dict = nothing)

Port of transformers' `CLIPModel`, the dual encoder of CLIP (Radford et al., 2021), built
from its `CLIPConfig`: the text encoder [`clip_text_model`](@ref
Metalhead.Transformers.clip_text_model) of `text_config` and the vision encoder
[`clip_vision_model`](@ref Metalhead.Transformers.clip_vision_model) of `vision_config`
(NamedTuples of the fields of their configs, or `nothing` for transformers' defaults, a
ViT-B/32), whose pooled features are projected to `projection_dim`. As in transformers,
`text_config_dict` and `vision_config_dict`, which older config.json files have, replace
`text_config` and `vision_config`: their fields over the defaults of the config classes.

See [`CLIPModel`](@ref Metalhead.Transformers.CLIPModel) for its inputs and outputs, and
[`CLIPProcessor`](@ref Metalhead.Transformers.CLIPProcessor), which
[`create_processor`](@ref Metalhead.create_processor) loads, for its inputs.
"""
function clip_model(; text_config = nothing, vision_config = nothing, projection_dim = 512,
        logit_scale_init_value = 2.6592, initializer_factor = 1.0,
        text_config_dict = nothing, vision_config_dict = nothing)
    text = something(text_config_dict, text_config, Some(nothing))
    vision = something(vision_config_dict, vision_config, Some(nothing))
    text_model = clip_text_model(; config_args(clip_text_model, text)...)
    vision_model = clip_vision_model(; config_args(clip_vision_model, vision)...)
    text_hidden = size(text_model.final_layer_norm.diag.scale, 1)
    vision_hidden = size(vision_model.post_layernorm.diag.scale, 1)
    return CLIPModel(text_model, vision_model,
        clip_projection(vision_hidden => projection_dim, initializer_factor),
        clip_projection(text_hidden => projection_dim, initializer_factor),
        init_array(() -> fill(Float32(logit_scale_init_value))))
end

# The `position_ids` buffers that checkpoints saved before transformers 4.31 hold
const CLIP_IGNORE = [r"(^|\.)embeddings\.position_ids$"]

# The main branch of openai/clip-vit-base-patch32 has pytorch_model.bin only: its commit is
# that of the pull request (refs/pr/66) of Hugging Face's conversion bot, which adds the
# same weights as model.safetensors, as transformers' `auto_conversion` would load them.
register!(ArchSpec(TRANSFORMERS, "CLIPModel", clip_model; family = "clip",
    ignore = CLIP_IGNORE,
    references = [
        "openai/clip-vit-base-patch32" => "c237dc49a33fc61debc9276459120b7eac67e7ef",
        "openai/clip-vit-large-patch14" => "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" =>
            "1a25a446712ba5ee05982a381eed697ef9b435cf"]))
# the text encoder of Stable Diffusion 1.5, that of openai/clip-vit-large-patch14
register!(ArchSpec(TRANSFORMERS, "CLIPTextModel", clip_text_model; family = "clip",
    ignore = CLIP_IGNORE,
    references = ["stable-diffusion-v1-5/stable-diffusion-v1-5/text_encoder" =>
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"]))
for (name, build) in ("CLIPTextModelWithProjection" => clip_text_model_with_projection,
    "CLIPVisionModel" => clip_vision_model,
    "CLIPVisionModelWithProjection" => clip_vision_model_with_projection)
    register!(ArchSpec(TRANSFORMERS, name, build; family = "clip", ignore = CLIP_IGNORE))
end
