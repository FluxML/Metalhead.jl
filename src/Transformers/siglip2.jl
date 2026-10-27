# Ported from transformers.models.siglip2.modeling_siglip2 (transformers 5.21, Apache-2.0,
# The Hugging Face Team): `Siglip2Model`, the image-text dual encoder of the NaFlex
# checkpoints of SigLIP 2, whose vision model takes images of any aspect ratio as patches,
# with the text model of SigLIP. Validated against test/fixtures/siglip2_* and the reference
# of google/siglip2-base-patch16-naflex.

@concrete struct Siglip2VisionEmbeddings
    patch_embedding
    position_embedding
end
Flux.@layer Siglip2VisionEmbeddings

# The number of patches (height, width) of each image, from `spatial_shapes` (2, batch).
function patch_grids(spatial_shapes, max_length)
    shapes = Array(spatial_shapes)
    size(shapes, 1) == 2 ||
        error("`spatial_shapes` has size $(size(shapes)), not (2, batch): a column (height, width) of patches for each image")
    grids = [(Int(h), Int(w)) for (h, w) in eachcol(shapes)]
    for (h, w) in grids
        h > 0 && w > 0 && h * w <= max_length ||
            error("an image of $h × $w patches (height × width, from `spatial_shapes`) doesn't fit in $max_length patches")
    end
    return grids
end
ChainRulesCore.@non_differentiable patch_grids(::Any...)

# transformers' `resize_positional_embeddings`: the position embeddings (hidden,
# num_patches) of a square grid, resized to the grid (h, w) of each image, bilinearly and
# antialiased, then padded to `max_length` with the first one, as (hidden, max_length,
# batch). transformers resizes in float32 on the CPU (and `Float16` input then promotes),
# and so does the port in at least `Float32`.
function resize_positional_embeddings(weight, spatial_shapes, max_length)
    d, n = size(weight)
    s = isqrt(n)
    T = promote_type(eltype(weight), Float32)
    grid = reshape(permutedims(reshape(T.(weight), d, s, s), (2, 3, 1)), s, s, d, 1)
    resized = map(patch_grids(spatial_shapes, max_length)) do (h, w)
        y = NNlib.upsample_bilinear(grid; size = (w, h), align_corners = false,
            antialias = true)
        e = eltype(weight).(permutedims(reshape(y, w * h, d), (2, 1)))
        h * w == max_length && return e
        return hcat(e, repeat(e[:, 1:1], 1, max_length - h * w))
    end
    return stack(resized)
end

function (m::Siglip2VisionEmbeddings)(pixel_values, spatial_shapes)
    x = m.patch_embedding(pixel_values)
    pos = resize_positional_embeddings(m.position_embedding.weight, spatial_shapes,
        size(x, 2))
    return x .+ pos
end

# The number of valid patches of each image, the first ones of `pixel_attention_mask`
# (num_patches, batch), as the image processor pads them; `nothing` without a mask.
function patch_lengths(mask, n, b)
    mask === nothing && return nothing
    size(mask) == (n, b) ||
        error("`pixel_attention_mask` has size $(size(mask)), the patches $((n, b))")
    valid = Array(mask) .!= 0
    lengths = vec(sum(valid; dims = 1))
    all(I -> valid[I] == (I[1] <= lengths[I[2]]), CartesianIndices(valid)) ||
        error("`pixel_attention_mask` must mark the first patches of each image: ones, then zeros, in each column")
    return lengths
end
ChainRulesCore.@non_differentiable patch_lengths(::Any...)

"""
    Siglip2VisionModel

transformers' `Siglip2VisionModel`, the vision encoder of the NaFlex checkpoints of
SigLIP 2, which [`siglip2_vision_model`](@ref Metalhead.Transformers.siglip2_vision_model)
builds.

    (model::Siglip2VisionModel)(pixel_values; spatial_shapes, pixel_attention_mask = nothing)
    (model::Siglip2VisionModel)(inputs::NamedTuple)

The `last_hidden_state` (hidden, num_patches, batch), after the final norm, and the
`pooler_output` (hidden, batch) that the attention-pooling `head` pools from it, of images
given as patches, as the image processor of a checkpoint makes them (a
[`Siglip2ImageProcessor`](@ref Metalhead.Transformers.Siglip2ImageProcessor), whose output
the model also takes as it is):

  - `pixel_values` (num_channels * patch_size^2, num_patches, batch): the patches of each
    image, row-major over the grid of the image, padded to `num_patches`, each flattened
    as a Julia array of size `(num_channels, patch_size, patch_size)` (channels first, then
    width);
  - `spatial_shapes` (2, batch): the numbers of patches `(height, width)` of each image, to
    which the position embeddings of a square grid are resized, bilinearly and
    antialiased, as in transformers;
  - `pixel_attention_mask` (num_patches, batch): 1 for the patches of the image, 0 for the
    padding after them, which no patch attends to (all patches are valid without it).
"""
@concrete struct Siglip2VisionModel
    embeddings
    encoder
    post_layernorm
    head
end
Flux.@layer Siglip2VisionModel

function (m::Siglip2VisionModel)(pixel_values; spatial_shapes,
        pixel_attention_mask = nothing)
    _, n, b = size(pixel_values)
    kv_lengths = patch_lengths(pixel_attention_mask, n, b)
    x = m.embeddings(pixel_values, spatial_shapes)
    x = m.post_layernorm(m.encoder(x, kv_lengths))
    return (; last_hidden_state = x, pooler_output = m.head(x, kv_lengths))
end

function (m::Siglip2VisionModel)(inputs::NamedTuple)
    check_inputs(m, inputs, (:pixel_values, :spatial_shapes);
        hint = "pass `images` to the processor")
    return m(inputs.pixel_values; inputs.spatial_shapes,
        pixel_attention_mask = get(inputs, :pixel_attention_mask, nothing))
end

"""
    Siglip2Model

transformers' `Siglip2Model`, the image-text dual encoder of the NaFlex checkpoints of
SigLIP 2 (Tschannen et al., 2025), such as `google/siglip2-base-patch16-naflex`, which
[`siglip2_model`](@ref Metalhead.Transformers.siglip2_model) builds: a
[`SiglipTextModel`](@ref Metalhead.Transformers.SiglipTextModel) `text_model`, a
[`Siglip2VisionModel`](@ref Metalhead.Transformers.Siglip2VisionModel) `vision_model`, which
takes images of any aspect ratio as patches, and the `logit_scale` and `logit_bias` of their
similarities. The SigLIP 2 checkpoints of fixed resolution
(`google/siglip2-base-patch16-224`) are [`SiglipModel`](@ref
Metalhead.Transformers.SiglipModel)s.

    (model::Siglip2Model)(input_ids, pixel_values; spatial_shapes,
                          pixel_attention_mask = nothing, attention_mask = nothing)
    (model::Siglip2Model)(inputs::NamedTuple)

The similarities of the texts `input_ids` (seq_len, num_texts), 1-based token indices, and
the images `pixel_values`, `spatial_shapes` and `pixel_attention_mask` (see
[`Siglip2VisionModel`](@ref Metalhead.Transformers.Siglip2VisionModel)), as a NamedTuple
with the keys of transformers' `Siglip2Output` (those of [`SiglipModel`](@ref
Metalhead.Transformers.SiglipModel)): `logits_per_image` (num_texts, num_images), whose
`sigmoid` is the probability that each text describes each image, `logits_per_text`, its
transpose, the normalized `text_embeds` and `image_embeds`, and the outputs of the two
models, `text_model_output` and `vision_model_output`. The model takes the output of the
processor of a checkpoint, a [`Siglip2Processor`](@ref
Metalhead.Transformers.Siglip2Processor), as it is (`model(inputs)`).

```julia
using Metalhead, Flux, HuggingFaceTokenizers
model = create_model("google/siglip2-base-patch16-naflex")
proc = create_processor("google/siglip2-base-patch16-naflex")
image = rand(UInt8, 640, 480, 3)                       # width × height × channels
inputs = proc(; images = image, text = ["a photo of a cat", "a photo of a dog"])
probs = sigmoid.(model(inputs).logits_per_image)       # 2 × 1
```
"""
@concrete struct Siglip2Model
    text_model
    vision_model
    logit_scale
    logit_bias
end
Flux.@layer Siglip2Model

function (m::Siglip2Model)(input_ids, pixel_values; spatial_shapes,
        pixel_attention_mask = nothing, attention_mask = nothing)
    vision = m.vision_model(pixel_values; spatial_shapes, pixel_attention_mask)
    return siglip_output(m, m.text_model(input_ids; attention_mask), vision)
end

function (m::Siglip2Model)(inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids, :pixel_values, :spatial_shapes);
        hint = SIGLIP_HINT)
    return m(inputs.input_ids, inputs.pixel_values; inputs.spatial_shapes,
        pixel_attention_mask = get(inputs, :pixel_attention_mask, nothing),
        attention_mask = get(inputs, :attention_mask, nothing))
end

function get_text_features(m::Siglip2Model, input_ids; attention_mask = nothing)
    return m.text_model(input_ids; attention_mask)
end

function get_text_features(m::Siglip2Model, inputs::NamedTuple)
    check_inputs(m, inputs, (:input_ids,); hint = "pass `text` to the processor")
    return get_text_features(m, inputs.input_ids;
        attention_mask = get(inputs, :attention_mask, nothing))
end

"""
    get_image_features(model::Siglip2Model, pixel_values; spatial_shapes,
                       pixel_attention_mask = nothing)
    get_image_features(model::Siglip2Model, inputs::NamedTuple)

The output of the vision model of SigLIP 2 on images given as patches (see
[`Siglip2VisionModel`](@ref Metalhead.Transformers.Siglip2VisionModel)), or on the output
of its image processor: `(; last_hidden_state, pooler_output)`, whose `pooler_output`
(hidden, num_images) are the embeddings of the images before their normalization, as
transformers' `get_image_features` returns them.
"""
function get_image_features(m::Siglip2Model, pixel_values; spatial_shapes,
        pixel_attention_mask = nothing)
    return m.vision_model(pixel_values; spatial_shapes, pixel_attention_mask)
end

function get_image_features(m::Siglip2Model, inputs::NamedTuple)
    check_inputs(m, inputs, (:pixel_values, :spatial_shapes);
        hint = "pass `images` to the processor")
    return m.vision_model(inputs)
end

"""
    siglip2_vision_model(; hidden_size = 768, intermediate_size = 3072,
        num_hidden_layers = 12, num_attention_heads = 12, num_channels = 3,
        patch_size = 16, num_patches = 256, hidden_act = "gelu_pytorch_tanh",
        layer_norm_eps = 1e-6, attention_dropout = 0.0, vision_use_head = true)

Port of transformers' `Siglip2VisionModel`, the vision encoder of the NaFlex checkpoints
of SigLIP 2 (see [`Siglip2VisionModel`](@ref Metalhead.Transformers.Siglip2VisionModel)),
built from the fields of its `Siglip2VisionConfig`: SigLIP's vision transformer, whose
patches, flattened, are embedded by a `Dense` layer, with `num_patches` learned position
embeddings of a square grid, resized to the grid of each image. `vision_use_head = false`
is not supported.
"""
function siglip2_vision_model(; hidden_size = 768, intermediate_size = 3072,
        num_hidden_layers = 12, num_attention_heads = 12, num_channels = 3,
        patch_size = 16, num_patches = 256, hidden_act = "gelu_pytorch_tanh",
        layer_norm_eps = 1e-6, attention_dropout = 0.0, vision_use_head = true)
    patch_size isa Integer ||
        error("`patch_size` must be an integer, got $(repr(patch_size))")
    isqrt(num_patches)^2 == num_patches ||
        error("`num_patches = $num_patches` must be a square, the position embeddings of a square grid")
    check_vision_use_head(vision_use_head)
    patch_dim = num_channels * patch_size^2
    embeddings = Siglip2VisionEmbeddings(
        Dense(patch_dim => hidden_size; init = fan_in_init(patch_dim)),
        Embedding(num_patches => hidden_size; init = fan_in_init(hidden_size)))
    encoder = siglip_encoder(; hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, hidden_act, layer_norm_eps, attention_dropout)
    head = siglip_pooling_head(; hidden_size, intermediate_size, num_attention_heads,
        hidden_act, layer_norm_eps)
    return Siglip2VisionModel(embeddings, encoder,
        LayerNorm(hidden_size; eps = Float32(layer_norm_eps)), head)
end

"""
    siglip2_model(; text_config = nothing, vision_config = nothing,
                  initializer_factor = 1.0)

Port of transformers' `Siglip2Model`, the image-text dual encoder of the NaFlex checkpoints
of SigLIP 2 (see [`Siglip2Model`](@ref Metalhead.Transformers.Siglip2Model)), built from its
`Siglip2Config`: the [`siglip_text_model`](@ref Metalhead.Transformers.siglip_text_model) of
`text_config` and the [`siglip2_vision_model`](@ref
Metalhead.Transformers.siglip2_vision_model) of `vision_config`, NamedTuples of the fields
of their configs (other keys are ignored), each `nothing` for the defaults of transformers.
The `projection_size` of the text model must be the `hidden_size` of the vision model.
"""
function siglip2_model(; text_config = nothing, vision_config = nothing,
        initializer_factor = 1.0)
    text_model, vision_model = siglip_towers(text_config, vision_config,
        siglip2_vision_model)
    return Siglip2Model(text_model, vision_model, zeros(Float32, 1), zeros(Float32, 1))
end

register!(ArchSpec(TRANSFORMERS, "Siglip2Model", siglip2_model; family = "siglip",
    references = ["google/siglip2-base-patch16-naflex" => "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"]))
MODEL_TYPE_CLASSES["siglip2"] = "Siglip2Model"
