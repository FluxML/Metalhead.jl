# Ported from transformers.models.modernbert.modeling_modernbert (transformers 5.20,
# Apache-2.0, Answer.AI, LightOn and The Hugging Face Team): `ModernBertModel`, and the heads
# `ModernBertForMaskedLM` and `ModernBertForSequenceClassification`. transformers pads the
# batches of its eager and SDPA attention and masks the padding, as this port does (the
# original code of Answer.AI unpads them for Flash Attention, with the same results at the
# tokens). Validated against test/fixtures/modernbert_* and the references of
# answerdotai/ModernBERT-base.

# `nn.LayerNorm(bias=False)`, which ModernBERT uses by default (`norm_bias = false`). Under
# `f16`/`bf16` its `weight` stays `Float32` and the output has the eltype of the input.
@concrete struct BiaslessLayerNorm
    weight
    eps
end
Flux.@layer BiaslessLayerNorm
Flux.f16(m::BiaslessLayerNorm, cx) = f32(m, cx)
Flux.bf16(m::BiaslessLayerNorm, cx) = f32(m, cx)

function (m::BiaslessLayerNorm)(x)
    bias = ChainRulesCore.ignore_derivatives(() -> zero(m.weight))
    return NNlib.layernorm(m.weight, bias, x; dims = 1, eps = m.eps)
end

@concrete struct ModernBertEmbeddings
    tok_embeddings
    norm
    drop
end
Flux.@layer ModernBertEmbeddings

(m::ModernBertEmbeddings)(input_ids) = m.drop(m.norm(m.tok_embeddings(input_ids)))

@concrete struct ModernBertMLP
    Wi
    act
    drop
    Wo
end
Flux.@layer ModernBertMLP

# A gated linear unit: `act(input) * gate` for the two halves of `Wi(x)`
(m::ModernBertMLP)(x) = m.Wo(m.drop(gated(m.act, m.Wi(x))))

# transformers' `ModernBertAttention`: self-attention over the keys within `sliding_window`
# of each query (`local_attention ÷ 2 + 1`, or `nothing` for global attention), with the
# rotary tables of the layer's type. It rotates in at least `Float32`.
@concrete struct ModernBertAttention
    layer_type
    head_dim
    sliding_window
    Wqkv
    Wo
    out_drop
end
Flux.@layer ModernBertAttention

function rotate_in_float32(x, cos, sin)
    T = promote_type(eltype(x), Float32)
    y = NNlib.apply_rotary_embedding(convert(AbstractArray{T}, x),
        convert(AbstractArray{T}, cos), convert(AbstractArray{T}, sin))
    return convert(AbstractArray{eltype(x)}, y)
end

function (m::ModernBertAttention)(x, position_embeddings, kv_lengths = nothing)
    _, n, b = size(x)
    cos, sin = position_embeddings[m.layer_type]
    qkv = reshape(m.Wqkv(x), m.head_dim, :, 3, n, b)
    q = rotate_in_float32(qkv[:, :, 1, :, :], cos, sin)
    k = rotate_in_float32(qkv[:, :, 2, :, :], cos, sin)
    v = qkv[:, :, 3, :, :]
    y = NNlib.scaled_dot_product_attention(q, k, v;
        fdrop = attention_fdrop(m.out_drop, q), kv_lengths,
        window = attention_window(n, m.sliding_window),
        q_lengths = sliding_window_q_lengths(kv_lengths, n, m.sliding_window))
    return m.out_drop(m.Wo(reshape(y, :, n, b)))
end

@concrete struct ModernBertEncoderLayer
    attn_norm
    attn
    mlp_norm
    mlp
end
Flux.@layer ModernBertEncoderLayer

function (m::ModernBertEncoderLayer)(x, position_embeddings, kv_lengths = nothing)
    x = x .+ m.attn(m.attn_norm(x), position_embeddings, kv_lengths)
    return x .+ m.mlp(m.mlp_norm(x))
end

"""
    ModernBertModel

transformers' `ModernBertModel`, the encoder of ModernBERT (Warner et al., 2024) and of the
models built on it (GTE-ModernBERT, ModernColBERT, ...), which
[`modernbert_model`](@ref Metalhead.Transformers.modernbert_model) builds. It is a
[`TextEncoder`](@ref Metalhead.Transformers.TextEncoder): see it for the inputs and outputs. It
takes no `token_type_ids` and has no pooler.

The tokens are embedded and normalised, and go through pre-norm layers with rotary position
embeddings, bias-free linear layers, a gated feed-forward network (GeGLU) and no norm before
the first attention. One layer in `global_attn_every_n_layers` (3) attends to all tokens,
with a rotary base of 160 000, and the others only to the tokens within `local_attention ÷ 2`
(64) of the query, with a base of 10 000.
"""
@concrete struct ModernBertModel <: TextEncoder
    embeddings
    layers
    final_norm
    rotary_emb
end
Flux.@layer ModernBertModel

function embed(m::ModernBertModel, input_ids; attention_mask = nothing)
    x = m.embeddings(input_ids)
    tables = rotary_tables(m.rotary_emb, x, size(input_ids, 1))
    return x, (tables, kv_lengths_of(attention_mask, input_ids))
end

encoder_layers(m::ModernBertModel) = m.layers
encoder_width(m::ModernBertModel) = size(m.embeddings.tok_embeddings.weight, 1)
layers_name(::ModernBertModel) = "layers"
encoder_prefix(::ModernBertModel) = ""
final_norm(m::ModernBertModel, x) = m.final_norm(x)
input_keys(::ModernBertModel) = (:attention_mask,)

@concrete struct ModernBertPredictionHead
    dense
    act
    norm
end
Flux.@layer ModernBertPredictionHead

(m::ModernBertPredictionHead)(x) = m.norm(m.act.(m.dense(x)))

"""
    ModernBertForMaskedLM

transformers' `ModernBertForMaskedLM`, a [`ModernBertModel`](@ref
Metalhead.Transformers.ModernBertModel) with the head of its masked language modeling, which
[`modernbert_for_masked_lm`](@ref Metalhead.Transformers.modernbert_for_masked_lm) builds,
and the architecture of `answerdotai/ModernBERT-base` and `-large`. Its `decoder` is tied to
the token embeddings.

    (model::ModernBertForMaskedLM)(input_ids; attention_mask = nothing)
    (model::ModernBertForMaskedLM)(inputs::NamedTuple)

The `logits` (vocab_size, seq_len, batch) of the token indices `input_ids`, of the tokens
that were masked or not. A vector is one sequence. [`forward_features`](@ref) returns the hidden
states of the base model.
"""
@concrete struct ModernBertForMaskedLM <: TextEncoderHead
    model
    head
    decoder
end
Flux.@layer ModernBertForMaskedLM

encoder_model(m::ModernBertForMaskedLM) = m.model
encoder_prefix(::ModernBertForMaskedLM) = "model."

function forward_head(m::ModernBertForMaskedLM, x; pre_logits::Bool = false)
    h = m.head(x)
    return pre_logits ? h : m.decoder(h)
end

function (m::ModernBertForMaskedLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    return forward_head(m, forward_features(m, input_ids; kwargs...))
end

"""
    ModernBertForSequenceClassification

transformers' `ModernBertForSequenceClassification`, a [`ModernBertModel`](@ref
Metalhead.Transformers.ModernBertModel) whose pooled hidden states (those of the first token,
or their mean over the tokens that `attention_mask` keeps, as `classifier_pooling` says) go
through a prediction head and a linear layer, as the rerankers built on ModernBERT do.
[`modernbert_for_sequence_classification`](@ref
Metalhead.Transformers.modernbert_for_sequence_classification) builds it.

    (model::ModernBertForSequenceClassification)(input_ids; attention_mask = nothing)
    (model::ModernBertForSequenceClassification)(inputs::NamedTuple)

The `logits` (num_labels, batch) of the token indices `input_ids` (seq_len, batch), or of
the output of the processor of the repository. A vector is one sequence, with logits
`(num_labels,)`. [`forward_features`](@ref) returns the hidden states of the base model.
"""
@concrete struct ModernBertForSequenceClassification <: TextEncoderHead
    model
    head
    drop
    classifier
    classifier_pooling
end
Flux.@layer ModernBertForSequenceClassification

encoder_model(m::ModernBertForSequenceClassification) = m.model
encoder_prefix(::ModernBertForSequenceClassification) = "model."

# The mean of the hidden states (hidden, seq_len, batch) of the tokens that `attention_mask`
# keeps, or of all
function masked_mean(x, attention_mask)
    attention_mask === nothing && return dropdims(sum(x; dims = 2); dims = 2) ./ size(x, 2)
    mask = mask_like(x, attention_mask)
    return dropdims(sum(x .* mask; dims = 2); dims = 2) ./
           dropdims(sum(mask; dims = 2); dims = 2)
end

# The `attention_mask` (seq_len, batch) as (1, seq_len, batch), of the type and on the device
# of `x`
function mask_like(x, attention_mask)
    mask = to_device_of(x, eltype(x).(Array(attention_mask)))
    return reshape(mask, 1, size(x, 2), :)
end
ChainRulesCore.@non_differentiable mask_like(::Any...)

function forward_head(m::ModernBertForSequenceClassification, x; attention_mask = nothing,
        pre_logits::Bool = false)
    pooled = m.classifier_pooling == "cls" ? first_token(x) : masked_mean(x, attention_mask)
    h = m.drop(m.head(pooled))
    return pre_logits ? h : m.classifier(h)
end

function (m::ModernBertForSequenceClassification)(input_ids::AbstractMatrix{<:Integer};
        attention_mask = nothing)
    x = forward_features(m, input_ids; attention_mask)
    return forward_head(m, x; attention_mask)
end

# Builders

# The rotary parameters of a layer type, from a dictionary or NamedTuple of the types'
# `rope_parameters`, the `rope_scaling` that updates both, and the base `theta` that the
# type lacks, as `ModernBertConfig.convert_rope_params_to_dict`.
function modernbert_rope(
        type, rope_parameters, rope_scaling, theta, max_position_embeddings)
    lookup(d, k) = d isa AbstractDict ? get(d, k, nothing) : get(d, Symbol(k), nothing)
    params = lookup(something(rope_parameters, (;)), type)
    params = merge(NamedTuple(Symbol(k) => v for (k, v) in pairs(something(params, (;)))),
        NamedTuple(Symbol(k) => v for (k, v) in pairs(something(rope_scaling, (;)))))
    return rope_type_parameters("ModernBertModel", params; rope_theta = theta,
        max_position_embeddings)
end

# transformers' `layer_types`: from the config, or one global layer in
# `global_attn_every_n_layers`, the first included
function modernbert_layer_types(layer_types, num_hidden_layers, global_attn_every_n_layers)
    layer_types === nothing || (layer_types = collect(String, layer_types))
    layer_types = something(layer_types,
        [i % global_attn_every_n_layers != 0 ? "sliding_attention" : "full_attention"
         for i in 0:(num_hidden_layers - 1)])
    length(layer_types) == num_hidden_layers ||
        error("`layer_types` has $(length(layer_types)) entries, `num_hidden_layers = $num_hidden_layers`")
    unknown = setdiff(layer_types, ["full_attention", "sliding_attention"])
    isempty(unknown) ||
        error("`layer_types` has $(repr(first(unknown))), not \"full_attention\" or \"sliding_attention\"")
    return layer_types
end

"""
    modernbert_model(; vocab_size = 50368, hidden_size = 768, intermediate_size = 1152,
        num_hidden_layers = 22, num_attention_heads = 12, hidden_activation = "gelu",
        max_position_embeddings = 8192, initializer_range = 0.02, norm_eps = 1e-5,
        norm_bias = false, attention_bias = false, attention_dropout = 0.0,
        layer_types = nothing, rope_parameters = nothing, local_attention = 128,
        embedding_dropout = 0.0, mlp_bias = false, mlp_dropout = 0.0, ...)

Port of transformers' `ModernBertModel` (see [`ModernBertModel`](@ref
Metalhead.Transformers.ModernBertModel) for its inputs and outputs), built from the fields of
its `ModernBertConfig` with the same names and defaults, and the keys of the older
configurations that it still reads: `global_attn_every_n_layers` (3), which sets the
`layer_types`, `global_rope_theta` (160 000) and `local_rope_theta` (10 000), which set the
`rope_theta` of the `rope_parameters` of the two layer types, and `rope_scaling`, which
updates both. The rotary embeddings are the default one, or `"linear"` or `"llama3"`.
"""
function modernbert_model(;
        vocab_size = 50368, hidden_size = 768, intermediate_size = 1152,
        num_hidden_layers = 22, num_attention_heads = 12, hidden_activation = "gelu",
        max_position_embeddings = 8192, initializer_range = 0.02, norm_eps = 1e-5,
        norm_bias = false, attention_bias = false, attention_dropout = 0.0,
        layer_types = nothing, rope_parameters = nothing, rope_scaling = nothing,
        global_rope_theta = nothing, local_rope_theta = nothing,
        global_attn_every_n_layers = 3, local_attention = 128, embedding_dropout = 0.0,
        mlp_bias = false, mlp_dropout = 0.0)
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size` ($hidden_size) must be divisible by `num_attention_heads` ($num_attention_heads)")
    layer_types = modernbert_layer_types(layer_types, num_hidden_layers,
        global_attn_every_n_layers)
    head_dim = hidden_size ÷ num_attention_heads
    types = sort!(unique(layer_types))
    thetas = Dict("full_attention" => something(global_rope_theta, 160_000.0),
        "sliding_attention" => something(local_rope_theta, 10_000.0))
    rotary_emb = NamedTuple(Symbol(t) => RotaryEmbedding(head_dim,
                                modernbert_rope(
                                    t, rope_parameters, rope_scaling, thetas[t],
                                    max_position_embeddings))
    for t in types)
    window = local_attention ÷ 2 + 1
    act = act_fn(hidden_activation; table = ENCODER_ACT2FN, key = "hidden_activation")
    init = normal_init(initializer_range)
    dense(io, bias) = init_layer(Dense, io; bias, init)
    function norm()
        norm_bias && return init_layer(LayerNorm, hidden_size; eps = Float32(norm_eps))
        return BiaslessLayerNorm(init_array(Flux.ones32, hidden_size; norm = true),
            Float64(norm_eps))
    end
    function layer(i)
        attn = ModernBertAttention(Symbol(layer_types[i + 1]), head_dim,
            layer_types[i + 1] == "sliding_attention" ? window : nothing,
            dense(hidden_size => 3hidden_size, attention_bias),
            dense(hidden_size => hidden_size, attention_bias),
            dropout_or_identity(attention_dropout))
        mlp = ModernBertMLP(dense(hidden_size => 2intermediate_size, mlp_bias), act,
            dropout_or_identity(mlp_dropout),
            dense(intermediate_size => hidden_size, mlp_bias))
        return ModernBertEncoderLayer(i == 0 ? identity : norm(), attn, norm(), mlp)
    end
    embeddings = ModernBertEmbeddings(
        init_layer(Embedding, vocab_size => hidden_size; init), norm(),
        dropout_or_identity(embedding_dropout))
    return ModernBertModel(embeddings, [layer(i) for i in 0:(num_hidden_layers - 1)],
        norm(), rotary_emb)
end

function modernbert_head(; hidden_size, initializer_range, classifier_bias,
        classifier_activation, norm_eps, norm_bias)
    init = normal_init(initializer_range)
    act = act_fn(classifier_activation; table = ENCODER_ACT2FN, key = "classifier_activation")
    norm = norm_bias ? init_layer(LayerNorm, hidden_size; eps = Float32(norm_eps)) :
           BiaslessLayerNorm(init_array(Flux.ones32, hidden_size; norm = true),
        Float64(norm_eps))
    return ModernBertPredictionHead(
        init_layer(Dense, hidden_size => hidden_size; bias = classifier_bias, init), act, norm)
end

"""
    modernbert_for_masked_lm(; ..., decoder_bias = true, classifier_bias = false,
        classifier_activation = "gelu", tie_word_embeddings = true)

Port of transformers' `ModernBertForMaskedLM` (see [`ModernBertForMaskedLM`](@ref
Metalhead.Transformers.ModernBertForMaskedLM)), built from the fields of its
`ModernBertConfig` (see [`modernbert_model`](@ref Metalhead.Transformers.modernbert_model))
with the same names and defaults. `sparse_prediction` only matters when training.
"""
function modernbert_for_masked_lm(; vocab_size = 50368, hidden_size = 768,
        intermediate_size = 1152, num_hidden_layers = 22, num_attention_heads = 12,
        hidden_activation = "gelu", max_position_embeddings = 8192, initializer_range = 0.02,
        norm_eps = 1e-5, norm_bias = false, attention_bias = false, attention_dropout = 0.0,
        layer_types = nothing, rope_parameters = nothing, rope_scaling = nothing,
        global_rope_theta = nothing, local_rope_theta = nothing,
        global_attn_every_n_layers = 3, local_attention = 128, embedding_dropout = 0.0,
        mlp_bias = false, mlp_dropout = 0.0, decoder_bias = true, classifier_bias = false,
        classifier_activation = "gelu", tie_word_embeddings = true)
    model = modernbert_model(; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads, hidden_activation, max_position_embeddings,
        initializer_range, norm_eps, norm_bias, attention_bias, attention_dropout,
        layer_types, rope_parameters, rope_scaling, global_rope_theta, local_rope_theta,
        global_attn_every_n_layers, local_attention, embedding_dropout, mlp_bias,
        mlp_dropout)
    head = modernbert_head(; hidden_size, initializer_range, classifier_bias,
        classifier_activation, norm_eps, norm_bias)
    bias = decoder_bias ? init_array(Flux.zeros32, vocab_size) : false
    decoder = if tie_word_embeddings
        TiedLinear(model.embeddings.tok_embeddings.weight, bias)
    else
        init_layer(Dense, hidden_size => vocab_size; bias = decoder_bias,
            init = normal_init(initializer_range))
    end
    return ModernBertForMaskedLM(model, head, decoder)
end

"""
    modernbert_for_sequence_classification(; ..., classifier_pooling = "cls",
        classifier_dropout = 0.0, classifier_bias = false, classifier_activation = "gelu",
        id2label = nothing, num_labels = nothing)

Port of transformers' `ModernBertForSequenceClassification` (see
[`ModernBertForSequenceClassification`](@ref
Metalhead.Transformers.ModernBertForSequenceClassification)), built from the fields of its
`ModernBertConfig` (see [`modernbert_model`](@ref Metalhead.Transformers.modernbert_model))
with the same names and defaults. The number of labels is the length of `id2label`, else
`num_labels`, else 2.
"""
function modernbert_for_sequence_classification(; vocab_size = 50368, hidden_size = 768,
        intermediate_size = 1152, num_hidden_layers = 22, num_attention_heads = 12,
        hidden_activation = "gelu", max_position_embeddings = 8192, initializer_range = 0.02,
        norm_eps = 1e-5, norm_bias = false, attention_bias = false, attention_dropout = 0.0,
        layer_types = nothing, rope_parameters = nothing, rope_scaling = nothing,
        global_rope_theta = nothing, local_rope_theta = nothing,
        global_attn_every_n_layers = 3, local_attention = 128, embedding_dropout = 0.0,
        mlp_bias = false, mlp_dropout = 0.0, classifier_pooling = "cls",
        classifier_dropout = 0.0, classifier_bias = false, classifier_activation = "gelu",
        id2label = nothing, num_labels = nothing)
    classifier_pooling in ("cls", "mean") ||
        error("`classifier_pooling = $(repr(classifier_pooling))` must be \"cls\" or \"mean\"")
    model = modernbert_model(; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads, hidden_activation, max_position_embeddings,
        initializer_range, norm_eps, norm_bias, attention_bias, attention_dropout,
        layer_types, rope_parameters, rope_scaling, global_rope_theta, local_rope_theta,
        global_attn_every_n_layers, local_attention, embedding_dropout, mlp_bias,
        mlp_dropout)
    head = modernbert_head(; hidden_size, initializer_range, classifier_bias,
        classifier_activation, norm_eps, norm_bias)
    classifier = init_layer(Dense, hidden_size => num_labels_of(id2label, num_labels);
        init = normal_init(initializer_range))
    return ModernBertForSequenceClassification(model, head,
        dropout_or_identity(classifier_dropout), classifier, classifier_pooling)
end

register!(ArchSpec(
    TRANSFORMERS, "ModernBertModel", modernbert_model; family = "modernbert",
    references = ["Alibaba-NLP/gte-modernbert-base" => "e7f32e3c00f91d699e8c43b53106206bcc72bb22"]))
register!(ArchSpec(TRANSFORMERS, "ModernBertForMaskedLM", modernbert_for_masked_lm;
    family = "modernbert",
    references = ["answerdotai/ModernBERT-base" => "8949b909ec900327062f0ebf497f51aef5e6f0c8"]))
register!(ArchSpec(TRANSFORMERS, "ModernBertForSequenceClassification",
    modernbert_for_sequence_classification; family = "modernbert",
    references = ["Alibaba-NLP/gte-reranker-modernbert-base" => "f7481e6055501a30fb19d090657df9ec1f79ab2c"]))
MODEL_TYPE_CLASSES["modernbert"] = "ModernBertModel"
