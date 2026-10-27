# Ported from transformers.models.siglip.modeling_siglip (transformers 5.21, Apache-2.0,
# Google AI and The Hugging Face Team): `SiglipModel`, the image-text dual encoder of SigLIP,
# with its text and vision models. Validated against test/fixtures/siglip_* and the
# references of google/siglip-base-patch16-224 and google/siglip2-base-patch16-224.

@concrete struct SiglipVisionEmbeddings
    patch_embedding
    position_embedding
end
Flux.@layer SiglipVisionEmbeddings

# transformers' `interpolate_pos_encoding`: the position embeddings (hidden, num_positions)
# of a square grid, resized bicubically to a grid of `w × h` patches, unless the images are
# square and have as many patches.
function interpolate_pos_embeddings(weight, w, h, square)
    d, n = size(weight)
    n == w * h && square && return weight
    s = isqrt(n)
    s^2 == n ||
        error("the $n position embeddings make no square grid, to interpolate them")
    grid = reshape(permutedims(reshape(weight, d, s, s), (2, 3, 1)), s, s, d, 1)
    resized = NNlib.upsample_bicubic(grid; size = (w, h), align_corners = false)
    return permutedims(reshape(resized, w * h, d), (2, 1))
end

function (m::SiglipVisionEmbeddings)(pixel_values; interpolate_pos_encoding::Bool = false)
    x = m.patch_embedding(pixel_values)
    w, h, d, b = size(x)
    x = permutedims(reshape(x, w * h, d, b), (2, 1, 3))
    pos = m.position_embedding.weight
    if interpolate_pos_encoding
        square = size(pixel_values, 1) == size(pixel_values, 2)
        pos = interpolate_pos_embeddings(pos, w, h, square)
    elseif size(pos, 2) != w * h
        error("the images have $w × $h patches, the position embeddings $(size(pos, 2)): resize them to the size of the model, or pass `interpolate_pos_encoding = true`")
    end
    return x .+ pos
end

@concrete struct SiglipTextEmbeddings
    token_embedding
    position_embedding
end
Flux.@layer SiglipTextEmbeddings

function (m::SiglipTextEmbeddings)(input_ids)
    n = size(input_ids, 1)
    max_positions = size(m.position_embedding.weight, 2)
    n <= max_positions ||
        error("the sequences have $n tokens, more than `max_position_embeddings = $max_positions`")
    return m.token_embedding(input_ids) .+ m.position_embedding.weight[:, 1:n]
end

@concrete struct SiglipAttention
    num_heads
    k_proj
    v_proj
    q_proj
    out_proj
    dropout
end
Flux.@layer SiglipAttention

# Self-attention over the first `kv_lengths` tokens of each sequence (all without).
function (m::SiglipAttention)(x, kv_lengths)
    _, n, b = size(x)
    heads(p) = reshape(p(x), :, m.num_heads, n, b)
    q, k, v = heads(m.q_proj), heads(m.k_proj), heads(m.v_proj)
    y = NNlib.scaled_dot_product_attention(q, k, v; kv_lengths,
        fdrop = attention_fdrop(m.dropout, q))
    return m.out_proj(reshape(y, :, n, b))
end

@concrete struct SiglipMLP
    activation_fn
    fc1
    fc2
end
Flux.@layer SiglipMLP

(m::SiglipMLP)(x) = m.fc2(m.activation_fn(m.fc1(x)))

@concrete struct SiglipEncoderLayer
    layer_norm1
    self_attn
    layer_norm2
    mlp
end
Flux.@layer SiglipEncoderLayer

function (m::SiglipEncoderLayer)(x, kv_lengths)
    x = x .+ m.self_attn(m.layer_norm1(x), kv_lengths)
    return x .+ m.mlp(m.layer_norm2(x))
end

@concrete struct SiglipEncoder
    layers
end
Flux.@layer SiglipEncoder

function (m::SiglipEncoder)(x, kv_lengths)
    for layer in m.layers
        x = layer(x, kv_lengths)
    end
    return x
end

"""
    MultiheadAttention

PyTorch's `nn.MultiheadAttention` with `batch_first = True`, keys and values of the size of
the queries and no extra biases, as the attention-pooling heads of SigLIP use it: the
queries, keys and values are projected by the blocks of `in_proj_weight`, of size
`(embed_dim, 3 * embed_dim)` (torch's `(3 * embed_dim, embed_dim)`), and `in_proj_bias`.
Called on `(query, key, value)` of size `(embed_dim, len, batch)`, with the keys of each
sequence limited to its first `kv_lengths`, it returns `(attn_output, nothing)`, as torch
returns the output and the attention weights, which aren't computed.
"""
@concrete struct MultiheadAttention
    num_heads
    in_proj_weight
    in_proj_bias
    out_proj
end
Flux.@layer MultiheadAttention

# The heads (head_dim, num_heads, len, batch) of the `i`th of the query, key and value
# projections of `x`.
function in_projection(m::MultiheadAttention, x, i)
    d = size(x, 1)
    r = ((i - 1) * d + 1):(i * d)
    p = transpose(m.in_proj_weight[:, r]) * reshape(x, d, :) .+ m.in_proj_bias[r]
    return reshape(p, :, m.num_heads, size(x, 2), size(x, 3))
end

function (m::MultiheadAttention)(query, key, value; kv_lengths = nothing)
    y = NNlib.scaled_dot_product_attention(in_projection(m, query, 1),
        in_projection(m, key, 2), in_projection(m, value, 3); kv_lengths)
    return m.out_proj(reshape(y, size(query))), nothing
end

@concrete struct SiglipMultiheadAttentionPoolingHead
    probe
    attention
    layernorm
    mlp
end
Flux.@layer SiglipMultiheadAttentionPoolingHead

# The features (hidden, batch) that a learned query `probe` pools from the tokens `x`
# (hidden, len, batch), of which it attends to the first `kv_lengths` (all without).
function (m::SiglipMultiheadAttentionPoolingHead)(x, kv_lengths = nothing)
    probe = repeat(m.probe, 1, 1, size(x, 3))
    h = first(m.attention(probe, x, x; kv_lengths))
    h = h .+ m.mlp(m.layernorm(h))
    return h[:, 1, :]
end

"""
    SiglipTextModel

transformers' `SiglipTextModel`, the text encoder of SigLIP and SigLIP 2, which
[`siglip_text_model`](@ref Metalhead.Transformers.siglip_text_model) builds.

    (model::SiglipTextModel)(input_ids; attention_mask = nothing)

The `last_hidden_state` (hidden, seq_len, batch), after the final norm, of the token
indices `input_ids` (seq_len, batch), 1-based, and the `pooler_output`
(projection_size, batch), the projection by `head` of the features of the last token,
which is padding in texts padded to the length of the model, as SigLIP was trained. Every
token attends to every other, but to the padding after the ones of `attention_mask`.
"""
@concrete struct SiglipTextModel
    embeddings
    encoder
    final_layer_norm
    head
end
Flux.@layer SiglipTextModel

function (m::SiglipTextModel)(input_ids::AbstractMatrix{<:Integer}; attention_mask = nothing)
    kv_lengths = kv_lengths_of(attention_mask, input_ids)
    x = m.final_layer_norm(m.encoder(m.embeddings(input_ids), kv_lengths))
    return (; last_hidden_state = x, pooler_output = m.head(x[:, end, :]))
end
function (m::SiglipTextModel)(input_ids::AbstractVector{<:Integer}; attention_mask = nothing)
    mask = attention_mask === nothing ? nothing : reshape(attention_mask, :, 1)
    return m(reshape(input_ids, :, 1); attention_mask = mask)
end

"""
    SiglipVisionModel

transformers' `SiglipVisionModel`, the vision encoder of SigLIP and of the SigLIP 2
checkpoints of fixed resolution, which [`siglip_vision_model`](@ref
Metalhead.Transformers.siglip_vision_model) builds.

    (model::SiglipVisionModel)(pixel_values; interpolate_pos_encoding = false)

The `last_hidden_state` (hidden, num_patches, batch), after the final norm, of the images
`pixel_values` (width, height, channels, batch), and the `pooler_output` (hidden, batch),
the features that the attention-pooling `head` pools from them. The images have the
`image_size` of the model, unless `interpolate_pos_encoding`, which resizes the position
embeddings bicubically to the grid of patches of the images, as transformers does.
"""
@concrete struct SiglipVisionModel
    embeddings
    encoder
    post_layernorm
    head
end
Flux.@layer SiglipVisionModel

function (m::SiglipVisionModel)(pixel_values; interpolate_pos_encoding::Bool = false)
    x = m.embeddings(pixel_values; interpolate_pos_encoding)
    x = m.post_layernorm(m.encoder(x, nothing))
    return (; last_hidden_state = x, pooler_output = m.head(x))
end

"""
    SiglipModel

transformers' `SiglipModel`, the image-text dual encoder of SigLIP (Zhai et al., 2023) and
of the SigLIP 2 checkpoints of fixed resolution (`google/siglip2-base-patch16-224`), which
[`siglip_model`](@ref Metalhead.Transformers.siglip_model) builds: a [`SiglipTextModel`](@ref
Metalhead.Transformers.SiglipTextModel) `text_model`, a [`SiglipVisionModel`](@ref
Metalhead.Transformers.SiglipVisionModel) `vision_model`, and the `logit_scale` and
`logit_bias` of their similarities.

    (model::SiglipModel)(input_ids, pixel_values; attention_mask = nothing,
                         interpolate_pos_encoding = false)
    (model::SiglipModel)(inputs::NamedTuple)

The similarities of the texts `input_ids` (seq_len, num_texts), 1-based token indices, and
the images `pixel_values` (width, height, channels, num_images), as a NamedTuple with the
keys of transformers' `SiglipOutput`, in Julia's layout (PyTorch's dimensions reversed):

  - `logits_per_image` (num_texts, num_images): the logit that each text describes each
    image, `exp(logit_scale) * cos + logit_bias` of the cosine similarity of their
    embeddings, whose `sigmoid` is the probability (SigLIP's sigmoid loss scores each pair
    alone, so the probabilities of the texts of an image don't sum to 1);
  - `logits_per_text` (num_images, num_texts), its transpose;
  - `text_embeds` (projection_size, num_texts) and `image_embeds` (hidden, num_images): the
    `pooler_output`s of the text and vision models, normalized to unit length;
  - `text_model_output` and `vision_model_output`: the outputs of the text and vision
    models, `(; last_hidden_state, pooler_output)`.

The texts are padded on the right to the length of the model (64 tokens), as SigLIP was
trained, and attend to the padding unless `attention_mask` excludes it; the processor of a
checkpoint, a [`SiglipProcessor`](@ref Metalhead.Transformers.SiglipProcessor), pads them
so and returns no mask, as transformers'. The model takes its output as it is
(`model(inputs)`). [`get_text_features`](@ref Metalhead.Transformers.get_text_features)
and [`get_image_features`](@ref Metalhead.Transformers.get_image_features) encode texts or
images alone.

```julia
using Metalhead, Flux, HuggingFaceTokenizers
model = create_model("google/siglip-base-patch16-224")
proc = create_processor("google/siglip-base-patch16-224")
image = rand(UInt8, 640, 480, 3)                       # width × height × channels
inputs = proc(; images = image, text = ["a photo of a cat", "a photo of a dog"])
probs = sigmoid.(model(inputs).logits_per_image)       # 2 × 1
```
"""
@concrete struct SiglipModel
    text_model
    vision_model
    logit_scale
    logit_bias
end
Flux.@layer SiglipModel

normalized(x) = x ./ sqrt.(sum(abs2, x; dims = 1))

# transformers' `SiglipOutput` of the outputs of the text and vision models.
function siglip_output(m, text_model_output, vision_model_output)
    text_embeds = normalized(text_model_output.pooler_output)
    image_embeds = normalized(vision_model_output.pooler_output)
    logits_per_image = transpose(text_embeds) * image_embeds .* exp.(m.logit_scale) .+
                       m.logit_bias
    return (; logits_per_image, logits_per_text = permutedims(logits_per_image),
        text_embeds, image_embeds, text_model_output, vision_model_output)
end

function (m::SiglipModel)(input_ids, pixel_values; attention_mask = nothing,
        interpolate_pos_encoding::Bool = false)
    vision = m.vision_model(pixel_values; interpolate_pos_encoding)
    return siglip_output(m, m.text_model(input_ids; attention_mask), vision)
end

const SIGLIP_HINT = "pass `images` and `text` to the processor, or encode them alone with `get_image_features` and `get_text_features`"

function (m::SiglipModel)(inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids, :pixel_values); hint = SIGLIP_HINT)
    return m(inputs.input_ids, inputs.pixel_values;
        attention_mask = get(inputs, :attention_mask, nothing))
end

"""
    get_text_features(model::Union{SiglipModel, Siglip2Model}, input_ids;
                      attention_mask = nothing)
    get_text_features(model::Union{SiglipModel, Siglip2Model}, inputs::NamedTuple)

The output of the text model of SigLIP or SigLIP 2 on the token indices `input_ids`
(seq_len, num_texts), 1-based, or on the `input_ids` (and `attention_mask`) of the output of
its processor: `(; last_hidden_state, pooler_output)` (see [`SiglipTextModel`](@ref
Metalhead.Transformers.SiglipTextModel)), whose `pooler_output` (projection_size,
num_texts) are the embeddings of the texts before their normalization, as transformers'
`get_text_features` returns them.
"""
function get_text_features(m::SiglipModel, input_ids; attention_mask = nothing)
    return m.text_model(input_ids; attention_mask)
end

function get_text_features(m::SiglipModel, inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids,); hint = "pass `text` to the processor")
    return get_text_features(m, inputs.input_ids;
        attention_mask = get(inputs, :attention_mask, nothing))
end

"""
    get_image_features(model::SiglipModel, pixel_values; interpolate_pos_encoding = false)
    get_image_features(model::SiglipModel, inputs::NamedTuple)

The output of the vision model of SigLIP on the images `pixel_values` (width, height,
channels, num_images), or on the `pixel_values` of the output of its processor:
`(; last_hidden_state, pooler_output)` (see [`SiglipVisionModel`](@ref
Metalhead.Transformers.SiglipVisionModel)), whose `pooler_output` (hidden, num_images) are
the embeddings of the images before their normalization, as transformers'
`get_image_features` returns them.
"""
function get_image_features(m::SiglipModel, pixel_values;
        interpolate_pos_encoding::Bool = false)
    return m.vision_model(pixel_values; interpolate_pos_encoding)
end

function get_image_features(m::SiglipModel, inputs::NamedTuple)
    check_inputs(m, inputs, (:pixel_values,); hint = "pass `images` to the processor")
    return get_image_features(m, inputs.pixel_values)
end

# Builders

const SIGLIP_ACT2FN = SAM3_ACT2FN

function siglip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout)
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size = $hidden_size` must be divisible by `num_attention_heads = $num_attention_heads`")
    act = act_fn(hidden_act; table = SIGLIP_ACT2FN)
    dense(io) = init_layer(Dense, io; init = Flux.glorot_uniform)
    norm() = init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps))
    layers = [SiglipEncoderLayer(norm(),
                  SiglipAttention(num_attention_heads, dense(hidden_size => hidden_size),
                      dense(hidden_size => hidden_size), dense(hidden_size => hidden_size),
                      dense(hidden_size => hidden_size),
                      dropout_or_identity(attention_dropout)),
                  norm(), siglip_mlp(act, hidden_size, intermediate_size))
              for _ in 1:num_hidden_layers]
    return SiglipEncoder(layers)
end

function siglip_mlp(act, hidden_size, intermediate_size)
    return SiglipMLP(act,
        init_layer(Dense, hidden_size => intermediate_size; init = Flux.glorot_uniform),
        init_layer(Dense, intermediate_size => hidden_size; init = Flux.glorot_uniform))
end

function siglip_pooling_head(; hidden_size, intermediate_size, num_attention_heads,
        hidden_act, layer_norm_eps)
    glorot = Flux.glorot_uniform
    attention = MultiheadAttention(num_attention_heads,
        init_array(glorot, hidden_size, 3hidden_size),
        init_array(Flux.zeros32, 3hidden_size),
        init_layer(Dense, hidden_size => hidden_size; init = glorot))
    return SiglipMultiheadAttentionPoolingHead(init_array(glorot, hidden_size, 1, 1),
        attention, init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps)),
        siglip_mlp(act_fn(hidden_act; table = SIGLIP_ACT2FN), hidden_size,
            intermediate_size))
end

# transformers' `default_flax_embed_init_` and `lecun_normal_`, without the truncation
fan_in_init(fan_in) = normal_init(1 / sqrt(fan_in))

"""
    siglip_text_model(; vocab_size = 32000, hidden_size = 768, intermediate_size = 3072,
        num_hidden_layers = 12, num_attention_heads = 12, max_position_embeddings = 64,
        hidden_act = "gelu_pytorch_tanh", layer_norm_eps = 1e-6, attention_dropout = 0.0,
        projection_size = nothing)

Port of transformers' `SiglipTextModel`, the text encoder of SigLIP and SigLIP 2 (see
[`SiglipTextModel`](@ref Metalhead.Transformers.SiglipTextModel)), built from the fields of
its `SiglipTextConfig` (or `Siglip2TextConfig`, the same): a transformer whose tokens attend
to each other in both directions, and whose last token is projected by `head` to
`projection_size` features (`hidden_size` by default).
"""
function siglip_text_model(; vocab_size = 32000, hidden_size = 768,
        intermediate_size = 3072, num_hidden_layers = 12, num_attention_heads = 12,
        max_position_embeddings = 64, hidden_act = "gelu_pytorch_tanh",
        layer_norm_eps = 1e-6, attention_dropout = 0.0, projection_size = nothing)
    projection_size = something(projection_size, hidden_size)
    embeddings = SiglipTextEmbeddings(
        init_layer(Embedding, vocab_size => hidden_size; init = fan_in_init(hidden_size)),
        init_layer(Embedding, max_position_embeddings => hidden_size;
            init = fan_in_init(hidden_size)))
    encoder = siglip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout)
    return SiglipTextModel(embeddings, encoder,
        init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps)),
        init_layer(Dense, hidden_size => projection_size; init = fan_in_init(hidden_size)))
end

function check_vision_use_head(vision_use_head)
    vision_use_head ||
        error("`vision_use_head = false` is not supported: the Metalhead ports of SigLIP's vision models have their attention-pooling head")
end

"""
    siglip_vision_model(; hidden_size = 768, intermediate_size = 3072,
        num_hidden_layers = 12, num_attention_heads = 12, num_channels = 3,
        image_size = 224, patch_size = 16, hidden_act = "gelu_pytorch_tanh",
        layer_norm_eps = 1e-6, attention_dropout = 0.0, vision_use_head = true)

Port of transformers' `SiglipVisionModel`, the vision encoder of SigLIP (see
[`SiglipVisionModel`](@ref Metalhead.Transformers.SiglipVisionModel)), built from the fields
of its `SiglipVisionConfig`: a vision transformer of the patches of `patch_size × patch_size`
pixels of images of `image_size × image_size`, with learned position embeddings and no
class token, whose tokens an attention-pooling head pools. `vision_use_head = false`, which
the configurations of some vision-language models carry, is not supported.
"""
function siglip_vision_model(; hidden_size = 768, intermediate_size = 3072,
        num_hidden_layers = 12, num_attention_heads = 12, num_channels = 3,
        image_size = 224, patch_size = 16, hidden_act = "gelu_pytorch_tanh",
        layer_norm_eps = 1e-6, attention_dropout = 0.0, vision_use_head = true)
    image_size isa Integer && patch_size isa Integer ||
        error("`image_size` and `patch_size` must be integers, as transformers' SiglipVisionEmbeddings requires; got $(repr(image_size)) and $(repr(patch_size))")
    check_vision_use_head(vision_use_head)
    num_positions = (image_size ÷ patch_size)^2
    embeddings = SiglipVisionEmbeddings(
        init_layer(Conv, (patch_size, patch_size), num_channels => hidden_size;
            stride = patch_size, init = fan_in_init(num_channels * patch_size^2)),
        init_layer(Embedding, num_positions => hidden_size;
            init = fan_in_init(hidden_size)))
    encoder = siglip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout)
    head = siglip_pooling_head(; hidden_size, intermediate_size, num_attention_heads,
        hidden_act, layer_norm_eps)
    return SiglipVisionModel(embeddings, encoder,
        init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps)), head)
end

# The text and vision models of the `text_config` and `vision_config` of a config, whose
# embeddings must have the same size.
function siglip_towers(text_config, vision_config, vision_builder)
    text_model = siglip_text_model(; config_args(siglip_text_model, text_config)...)
    vision_model = vision_builder(; config_args(vision_builder, vision_config)...)
    text_dim = size(text_model.head.weight, 1)
    image_dim = size(vision_model.post_layernorm.diag.scale, 1)
    text_dim == image_dim ||
        error("the texts are embedded in $text_dim features (`projection_size` of `text_config`), the images in $image_dim (`hidden_size` of `vision_config`): they must be equal")
    return text_model, vision_model
end

"""
    siglip_model(; text_config = nothing, vision_config = nothing, initializer_factor = 1.0)

Port of transformers' `SiglipModel`, the image-text dual encoder of SigLIP and of the
SigLIP 2 checkpoints of fixed resolution (see [`SiglipModel`](@ref
Metalhead.Transformers.SiglipModel)), built from its `SiglipConfig`: the
[`siglip_text_model`](@ref Metalhead.Transformers.siglip_text_model) of `text_config` and
the [`siglip_vision_model`](@ref Metalhead.Transformers.siglip_vision_model) of
`vision_config`, NamedTuples of the fields of their configs (other keys are ignored), each
`nothing` for the defaults of transformers. The `projection_size` of the text model must be
the `hidden_size` of the vision model.
"""
function siglip_model(; text_config = nothing, vision_config = nothing,
        initializer_factor = 1.0)
    text_model, vision_model = siglip_towers(text_config, vision_config,
        siglip_vision_model)
    return SiglipModel(text_model, vision_model, init_array(Flux.zeros32, 1),
        init_array(Flux.zeros32, 1))
end

register!(ArchSpec(TRANSFORMERS, "SiglipModel", siglip_model; family = "siglip",
    references = ["google/siglip-base-patch16-224" => "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "google/siglip2-base-patch16-224" => "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"]))
MODEL_TYPE_CLASSES["siglip"] = "SiglipModel"
