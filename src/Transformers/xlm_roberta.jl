# Ported from transformers.models.xlm_roberta.modeling_xlm_roberta (transformers 5.20,
# Apache-2.0, Facebook AI Research and The Hugging Face Team): the encoder `XLMRobertaModel`
# and `XLMRobertaForSequenceClassification`, without the decoder (`is_decoder`, cross-attention
# and the key-value cache). XLM-RoBERTa is Roberta with another vocabulary, and has the layers
# of BERT (`bert.jl`); only its embeddings and its heads differ. Validated against
# test/fixtures/xlm_roberta_* and the references of BAAI/bge-reranker-base and
# BAAI/bge-reranker-v2-m3.

# The 1-based indices in the position embeddings of the tokens `input_ids` (1-based): each token
# that is not the padding one has its position among the tokens that are not, counting from
# `padding_idx + 1`, and the padding has the position `padding_idx` (transformers'
# `create_position_ids_from_input_ids`, taken from fairseq).
function position_indices(input_ids, padding_idx)
    mask = eltype(input_ids).(input_ids .!= padding_idx + 1)
    return cumsum(mask; dims = 1) .* mask .+ (padding_idx + 1)
end
ChainRulesCore.@non_differentiable position_indices(::Any...)

# transformers' `XLMRobertaEmbeddings`: those of BERT, with positions numbered after the
# padding.
@concrete struct XLMRobertaEmbeddings
    word_embeddings
    token_type_embeddings
    LayerNorm
    dropout
    position_embeddings
    padding_idx
end
Flux.@layer XLMRobertaEmbeddings

# An error for positions past the `max_positions` that the model embeds
function check_xlm_roberta_positions(positions, max_positions, padding_idx)
    maximum(positions; init = 0) <= max_positions ||
        error("the sequences have too many tokens: XLM-RoBERTa numbers the positions from `pad_token_id + 1 = $(padding_idx + 1)`, so `max_position_embeddings = $max_positions` holds up to $(max_positions - padding_idx - 1) tokens")
    return nothing
end
ChainRulesCore.@non_differentiable check_xlm_roberta_positions(::Any...)

function (m::XLMRobertaEmbeddings)(input_ids, token_type_ids = nothing)
    positions = position_indices(input_ids, m.padding_idx)
    check_xlm_roberta_positions(positions, size(m.position_embeddings.weight, 2),
        m.padding_idx)
    x = m.word_embeddings(input_ids) .+
        token_type_features(m.token_type_embeddings, token_type_ids, input_ids)
    x = x .+ m.position_embeddings(positions)
    return m.dropout(m.LayerNorm(x))
end

"""
    XLMRobertaModel

transformers' `XLMRobertaModel`, the multilingual encoder of XLM-RoBERTa (Conneau et al., 2019)
and of the models built on it (BGE-M3 and the BGE rerankers, ...), which
[`xlm_roberta_model`](@ref Metalhead.Transformers.xlm_roberta_model) builds. It is a
[`TextEncoder`](@ref Metalhead.Transformers.TextEncoder): see it for the inputs and outputs.

It has the layers of [`BertModel`](@ref Metalhead.Transformers.BertModel) and differs in its
position embeddings: the positions count the tokens that are not the padding token
(`pad_token_id`, 1), from `pad_token_id + 1`, and the padding has the position `pad_token_id`.
They follow the token indices, not the `attention_mask`. A sequence holds at most
`max_position_embeddings - pad_token_id - 1` tokens (512 of the 514 of XLM-RoBERTa).
"""
@concrete struct XLMRobertaModel <: BertLikeModel
    embeddings
    encoder
    pooler
end
Flux.@layer XLMRobertaModel

encoder_prefix(::XLMRobertaModel) = ""

@concrete struct XLMRobertaClassificationHead
    dense
    dropout
    out_proj
end
Flux.@layer XLMRobertaClassificationHead

function (m::XLMRobertaClassificationHead)(features; pre_logits::Bool = false)
    x = m.dropout(tanh.(m.dense(m.dropout(first_token(features)))))
    return pre_logits ? x : m.out_proj(x)
end

"""
    XLMRobertaForSequenceClassification

transformers' `XLMRobertaForSequenceClassification`, an [`XLMRobertaModel`](@ref
Metalhead.Transformers.XLMRobertaModel) without a pooler, with a classification head on the
hidden states of its first token (`<s>`): a dense layer, `tanh` and a linear layer. The BGE
rerankers are such models, with one label, the relevance of the second text of a pair to the
first. [`xlm_roberta_for_sequence_classification`](@ref
Metalhead.Transformers.xlm_roberta_for_sequence_classification) builds it.

    (model::XLMRobertaForSequenceClassification)(input_ids; attention_mask = nothing,
                                                 token_type_ids = nothing)
    (model::XLMRobertaForSequenceClassification)(inputs::NamedTuple)

The `logits` (num_labels, batch) of the token indices `input_ids` (seq_len, batch), or of
the output of the processor of the repository. A vector is one sequence, with logits
`(num_labels,)`. [`forward_features`](@ref) returns the hidden states of the base model.
"""
@concrete struct XLMRobertaForSequenceClassification <: TextEncoderHead
    classifier
    roberta
end
Flux.@layer XLMRobertaForSequenceClassification

encoder_model(m::XLMRobertaForSequenceClassification) = m.roberta
encoder_prefix(::XLMRobertaForSequenceClassification) = "roberta."

function forward_head(m::XLMRobertaForSequenceClassification, x; pre_logits::Bool = false)
    return m.classifier(x; pre_logits)
end

function (m::XLMRobertaForSequenceClassification)(input_ids::AbstractMatrix{<:Integer};
        kwargs...)
    return forward_head(m, forward_features(m, input_ids; kwargs...))
end

# Builders

"""
    xlm_roberta_model(; vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12,
        num_attention_heads = 12, intermediate_size = 3072, hidden_act = "gelu",
        hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 512, type_vocab_size = 2, initializer_range = 0.02,
        layer_norm_eps = 1e-12, pad_token_id = 1, add_pooling_layer = true)

Port of transformers' `XLMRobertaModel` (see [`XLMRobertaModel`](@ref
Metalhead.Transformers.XLMRobertaModel) for its inputs and outputs), built from the fields of
its `XLMRobertaConfig` with the same names and defaults (the checkpoints set
`max_position_embeddings = 514` and `type_vocab_size = 1`). See [`bert_model`](@ref
Metalhead.Transformers.bert_model) for the activations and the options. The decoder is not
supported.
"""
function xlm_roberta_model(; vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12,
        num_attention_heads = 12, intermediate_size = 3072, hidden_act = "gelu",
        hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 512, type_vocab_size = 2, initializer_range = 0.02,
        layer_norm_eps = 1e-12, pad_token_id = 1, is_decoder = false,
        add_cross_attention = false, add_pooling_layer = true)
    check_encoder("XLMRobertaModel"; is_decoder, add_cross_attention)
    pad_token_id isa Integer ||
        error("`pad_token_id` must be an integer: XLM-RoBERTa numbers the positions after it")
    init = normal_init(initializer_range)
    embedding(n) = init_layer(Embedding, n => hidden_size; init)
    function embeddings(norm, dropout)
        return XLMRobertaEmbeddings(embedding(vocab_size), embedding(type_vocab_size),
            norm(), dropout(), embedding(max_position_embeddings), pad_token_id)
    end
    return bert_like(XLMRobertaModel, embeddings; hidden_size, num_hidden_layers,
        num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob,
        attention_probs_dropout_prob, initializer_range, layer_norm_eps, add_pooling_layer)
end

"""
    xlm_roberta_for_sequence_classification(; vocab_size = 30522, hidden_size = 768, ...,
        classifier_dropout = nothing, id2label = nothing, num_labels = nothing)

Port of transformers' `XLMRobertaForSequenceClassification` (see
[`XLMRobertaForSequenceClassification`](@ref
Metalhead.Transformers.XLMRobertaForSequenceClassification)), built from the fields of its
`XLMRobertaConfig` (see [`xlm_roberta_model`](@ref Metalhead.Transformers.xlm_roberta_model))
with the same names and defaults. The number of labels is the length of `id2label`, else
`num_labels`, else 2; `classifier_dropout` defaults to `hidden_dropout_prob`.
"""
function xlm_roberta_for_sequence_classification(; vocab_size = 30522, hidden_size = 768,
        num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072,
        hidden_act = "gelu", hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 512, type_vocab_size = 2, initializer_range = 0.02,
        layer_norm_eps = 1e-12, pad_token_id = 1, is_decoder = false,
        add_cross_attention = false, classifier_dropout = nothing, id2label = nothing,
        num_labels = nothing)
    roberta = xlm_roberta_model(; vocab_size, hidden_size, num_hidden_layers,
        num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob,
        attention_probs_dropout_prob, max_position_embeddings, type_vocab_size,
        initializer_range, layer_norm_eps, pad_token_id, is_decoder, add_cross_attention,
        add_pooling_layer = false)
    init = normal_init(initializer_range)
    dense(io) = init_layer(Dense, io; init)
    dropout() = dropout_or_identity(something(classifier_dropout, hidden_dropout_prob))
    classifier = XLMRobertaClassificationHead(dense(hidden_size => hidden_size), dropout(),
        dense(hidden_size => num_labels_of(id2label, num_labels)))
    return XLMRobertaForSequenceClassification(classifier, roberta)
end

register!(ArchSpec(TRANSFORMERS, "XLMRobertaModel", xlm_roberta_model;
    family = "xlm_roberta", ignore = POSITION_IDS_IGNORE,
    references = ["intfloat/multilingual-e5-base" => "d128750597153bb5987e10b1c3493a34e5a4502a"]))
register!(ArchSpec(TRANSFORMERS, "XLMRobertaForSequenceClassification",
    xlm_roberta_for_sequence_classification; family = "xlm_roberta",
    ignore = POSITION_IDS_IGNORE,
    references = ["BAAI/bge-reranker-base" => "2cfc18c9415c912f9d8155881c133215df768a70",
        "BAAI/bge-reranker-v2-m3" => "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"]))
MODEL_TYPE_CLASSES["xlm-roberta"] = "XLMRobertaModel"
