# Ported from transformers.models.bert.modeling_bert (transformers 5.20, Apache-2.0, Google AI
# Language Team, NVIDIA and The Hugging Face Team): the encoder `BertModel` and
# `BertForSequenceClassification`, without the decoder (`is_decoder`, cross-attention and the
# key-value cache). Validated against test/fixtures/bert_* and the references of
# sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5 and
# cross-encoder/ms-marco-MiniLM-L6-v2.

# The activations of the encoder configs, by name (transformers' `ACT2FN`). "gelu_new" and
# "gelu_pytorch_tanh" are the same tanh approximation.
const ENCODER_ACT2FN = Dict("gelu" => gelu_erf, "gelu_new" => gelu_tanh,
    "gelu_pytorch_tanh" => gelu_tanh, "relu" => relu, "silu" => swish, "swish" => swish,
    "quick_gelu" => quick_gelu)

"""
    TextEncoder

The base models of the text encoders BERT, XLM-RoBERTa and ModernBERT, which map token
indices `input_ids` (seq_len, batch), 1-based, to the hidden states of each token:
[`BertModel`](@ref Metalhead.Transformers.BertModel),
[`XLMRobertaModel`](@ref Metalhead.Transformers.XLMRobertaModel) and
[`ModernBertModel`](@ref Metalhead.Transformers.ModernBertModel).

    (model::TextEncoder)(input_ids; attention_mask = nothing, token_type_ids = nothing)
    (model::TextEncoder)(inputs::NamedTuple)

The `last_hidden_state` of transformers' model, of size `(hidden_size, seq_len, batch)`, the
hidden states of each token after the last layer (and the final norm of ModernBERT), as the
decoder models return theirs. Every token attends to every other, in both directions. A
batch of sequences of different lengths, padded on the right, takes the tokenizer's
`attention_mask` (seq_len, batch), which keeps the padding out of the keys; the outputs at
the positions of the padding are not meaningful. `token_type_ids` (seq_len, batch) are the
segment ids of transformers, 0 or 1, and default to 0 (ModernBERT takes none). A vector
`input_ids` is a single sequence, whose output has no batch dimension. The output of the
processor of a repository, a [`TextProcessor`](@ref Metalhead.TextProcessor), can replace
all of them: `model(proc(texts))`.

[`forward_features`](@ref) is the same call, and [`forward_intermediates`](@ref) returns the
output of each layer (transformers' `hidden_states` without the embeddings). The BERT and
XLM-RoBERTa models also have transformers' `pooler_output` (hidden_size, batch), the first
token through the dense layer and `tanh` of the pooler: `forward_head(model, hidden_states)`,
which errors for a model built without a pooler (`add_pooling_layer = false`).
"""
abstract type TextEncoder end

# A text encoder with a head on top, built around the base model `encoder_model(model)`.
abstract type TextEncoderHead end

encoder_model(m::TextEncoder) = m

# The hidden states (hidden_size, seq_len, batch) after the last layer (and the final norm
# of ModernBERT), of the token indices `input_ids` (seq_len, batch).
function hidden_states(m::TextEncoder, input_ids::AbstractMatrix{<:Integer}; kwargs...)
    x, args = embed(m, input_ids; kwargs...)
    for layer in encoder_layers(m)
        x = layer(x, args...)
    end
    return final_norm(m, x)
end

final_norm(m::TextEncoder, x) = x

function (m::TextEncoder)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    return hidden_states(m, input_ids; kwargs...)
end

# A single sequence is a batch of one without its dimension
function (m::Union{TextEncoder, TextEncoderHead})(input_ids::AbstractVector{<:Integer};
        kwargs...)
    return unbatch(m(reshape(input_ids, :, 1); sequence_kwargs(kwargs)...))
end
unbatch(x::AbstractArray) = dropdims(x; dims = ndims(x))
# The keyword arguments of a single sequence as those of a batch of one: its masks and types
function sequence_kwargs(kwargs)
    one_sequence(k, v) = k in (:attention_mask, :token_type_ids) && v isa AbstractVector ?
                         reshape(v, :, 1) : v
    return NamedTuple(k => one_sequence(k, v) for (k, v) in kwargs)
end

function forward_features(m::Union{TextEncoder, TextEncoderHead},
        input_ids::AbstractMatrix{<:Integer}; kwargs...)
    return hidden_states(encoder_model(m), input_ids; kwargs...)
end
function forward_features(m::Union{TextEncoder, TextEncoderHead},
        input_ids::AbstractVector{<:Integer}; kwargs...)
    return unbatch(forward_features(m, reshape(input_ids, :, 1); sequence_kwargs(kwargs)...))
end

# Keyword arguments of the models that the output of a processor can hold
input_keys(::TextEncoder) = (:attention_mask, :token_type_ids)
input_keys(m::TextEncoderHead) = input_keys(encoder_model(m))
input_hint(::Union{TextEncoder, TextEncoderHead}) = "pass `text` to the processor"

function (m::Union{TextEncoder, TextEncoderHead})(inputs::NamedTuple)
    input_ids, kwargs = token_inputs(m, inputs)
    return m(input_ids; kwargs...)
end

function forward_features(m::Union{TextEncoder, TextEncoderHead}, inputs::NamedTuple)
    input_ids, kwargs = token_inputs(m, inputs)
    return forward_features(m, input_ids; kwargs...)
end

function forward_intermediates(m::Union{TextEncoder, TextEncoderHead}, inputs::NamedTuple;
        kwargs...)
    input_ids, model_kwargs = token_inputs(m, inputs)
    return forward_intermediates(m, input_ids; model_kwargs..., kwargs...)
end

"""
    forward_intermediates(model::Union{BertModel, XLMRobertaModel, ModernBertModel},
                          input_ids; indices = nothing, norm = false, stop_early = false,
                          intermediates_only = false, attention_mask = nothing,
                          token_type_ids = nothing)

The outputs of the layers of a text encoder (or of the model of a head such as
`BertForSequenceClassification`) for the token indices `input_ids` of size
`(seq_len, batch_size)`, each of size `(hidden_size, seq_len, batch_size)`, or of a vector for
a single sequence, each of size `(hidden_size, seq_len)`: the `i`-th intermediate is the
output of the `i`-th layer, transformers' `output_hidden_states[i]` (the embeddings are
`output_hidden_states[0]`), except that ModernBERT's last one is not normed unless
`norm = true`. The features are the `last_hidden_state`, as [`forward_features`](@ref)
returns them. `indices`, `norm`, `stop_early` and `intermediates_only` are those of the
vision models. The output of a processor can replace `input_ids` and the other
keyword arguments, as for the model. [`feature_info`](@ref) gives the `hidden_size` and the
module of each layer.
"""
function forward_intermediates(m::Union{TextEncoder, TextEncoderHead},
        input_ids::AbstractMatrix{<:Integer}; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, intermediates_only::Bool = false, kwargs...)
    model = encoder_model(m)
    layers = encoder_layers(model)
    take, max_index = feature_take_indices(length(layers), indices)
    x, args = embed(model, input_ids; kwargs...)
    stages = [h -> layer(h, args...) for layer in layers]
    outs = stage_outputs(stages, x, stop_early ? max_index : length(stages))
    intermediates = [norm ? final_norm(model, outs[i]) : outs[i] for i in take]
    intermediates_only && return intermediates
    return final_norm(model, last(outs)), intermediates
end

function forward_intermediates(m::Union{TextEncoder, TextEncoderHead},
        input_ids::AbstractVector{<:Integer}; intermediates_only::Bool = false, kwargs...)
    batched = forward_intermediates(m, reshape(input_ids, :, 1); intermediates_only,
        sequence_kwargs(kwargs)...)
    intermediates_only && return map(unbatch, batched)
    features, intermediates = batched
    return unbatch(features), map(unbatch, intermediates)
end

function feature_info(m::Union{TextEncoder, TextEncoderHead})
    model = encoder_model(m)
    prefix = encoder_prefix(m)
    return [(; num_chs = encoder_width(model), reduction = 1,
                module_name = prefix * layers_name(model) * ".$(i - 1)")
            for i in eachindex(encoder_layers(model))]
end

# The first token of each sequence of hidden states (hidden_size, seq_len, batch...)
first_token(x) = x[:, 1, ntuple(Returns(:), ndims(x) - 2)...]

# An error for sequences of `n` tokens, past the `max_positions` that the model embeds
function check_positions(n, max_positions)
    n <= max_positions ||
        error("the sequences have $n tokens, more than `max_position_embeddings = $max_positions`")
    return nothing
end

# transformers' `BertEmbeddings`: the sum of the word, position and token type embeddings
# of the tokens, normalised.
@concrete struct BertEmbeddings
    word_embeddings
    position_embeddings
    token_type_embeddings
    LayerNorm
    dropout
end
Flux.@layer BertEmbeddings

# The embeddings of the token types of `token_type_ids` (0-based), or of the type 0.
function token_type_features(embedding, token_type_ids, input_ids)
    token_type_ids === nothing && return embedding.weight[:, 1]
    size(token_type_ids) == size(input_ids) ||
        error("`token_type_ids` has size $(size(token_type_ids)), the token indices $(size(input_ids))")
    return embedding(token_type_ids .+ 1)
end

function (m::BertEmbeddings)(input_ids, token_type_ids = nothing)
    n = size(input_ids, 1)
    check_positions(n, size(m.position_embeddings.weight, 2))
    x = m.word_embeddings(input_ids) .+
        token_type_features(m.token_type_embeddings, token_type_ids, input_ids)
    x = x .+ m.position_embeddings.weight[:, 1:n]
    return m.dropout(m.LayerNorm(x))
end

@concrete struct BertSelfAttention
    num_attention_heads
    query
    key
    value
    dropout
end
Flux.@layer BertSelfAttention

function (m::BertSelfAttention)(x, kv_lengths)
    _, n, b = size(x)
    heads(p) = reshape(p(x), :, m.num_attention_heads, n, b)
    q, k, v = heads(m.query), heads(m.key), heads(m.value)
    y = NNlib.scaled_dot_product_attention(q, k, v; kv_lengths,
        fdrop = attention_fdrop(m.dropout, q))
    return reshape(y, :, n, b)
end

# `BertSelfOutput` and `BertOutput`: the projection of a sublayer, added to its input and
# normalised.
@concrete struct BertSelfOutput
    dense
    LayerNorm
    dropout
end
Flux.@layer BertSelfOutput

(m::BertSelfOutput)(h, residual) = m.LayerNorm(m.dropout(m.dense(h)) .+ residual)

@concrete struct BertAttention
    self
    output
end
Flux.@layer BertAttention

function (m::BertAttention)(x, kv_lengths)
    return m.output(m.self(x, kv_lengths), x)
end

@concrete struct BertIntermediate
    dense
    intermediate_act_fn
end
Flux.@layer BertIntermediate

(m::BertIntermediate)(x) = m.intermediate_act_fn(m.dense(x))

@concrete struct BertOutput
    dense
    LayerNorm
    dropout
end
Flux.@layer BertOutput

(m::BertOutput)(h, residual) = m.LayerNorm(m.dropout(m.dense(h)) .+ residual)

@concrete struct BertLayer
    attention
    intermediate
    output
end
Flux.@layer BertLayer

function (m::BertLayer)(x, kv_lengths = nothing)
    attention_output = m.attention(x, kv_lengths)
    return m.output(m.intermediate(attention_output), attention_output)
end

@concrete struct BertEncoder
    layer
end
Flux.@layer BertEncoder

# The base models with the layers of BERT: `BertModel` and `XLMRobertaModel`, which differ in
# their embeddings only.
abstract type BertLikeModel <: TextEncoder end

function embed(m::BertLikeModel, input_ids; attention_mask = nothing,
        token_type_ids = nothing)
    x = m.embeddings(input_ids, token_type_ids)
    return x, (kv_lengths_of(attention_mask, input_ids),)
end

encoder_layers(m::BertLikeModel) = m.encoder.layer
encoder_width(m::BertLikeModel) = size(m.embeddings.word_embeddings.weight, 1)
layers_name(::BertLikeModel) = "encoder.layer"

# transformers' `pooler_output` of the hidden states `x`
function forward_head(m::BertLikeModel, x)
    m.pooler === nothing &&
        error("$(nameof(typeof(m))) was built without a pooler (`add_pooling_layer = false`)")
    return m.pooler(x)
end

"""
    BertPooler

The pooler of BERT and XLM-RoBERTa's base models: the first token of the hidden states, through
a dense layer and `tanh`.
"""
@concrete struct BertPooler
    dense
    activation
end
Flux.@layer BertPooler

(m::BertPooler)(x) = m.activation.(m.dense(first_token(x)))

"""
    BertModel

transformers' `BertModel`, the encoder of BERT (Devlin et al., 2018) and of the many
models that share its architecture (MiniLM, the BGE and GTE embedding models, ELECTRA's and
cross-encoders' backbones, ...), which [`bert_model`](@ref Metalhead.Transformers.bert_model)
builds. It is a [`TextEncoder`](@ref Metalhead.Transformers.TextEncoder): see it for the
inputs and outputs.

The tokens are embedded by the sum of their word, absolute position and token type
embeddings, normalised, and go through `num_hidden_layers` post-norm layers of
bidirectional self-attention and a feed-forward network.

```julia
using Metalhead, HuggingFaceTokenizers
model = create_model(\"sentence-transformers/all-MiniLM-L6-v2\")
proc = create_processor(\"sentence-transformers/all-MiniLM-L6-v2\")
y = model(proc([\"A cat sits on the mat.\", \"Dogs bark.\"]))   # the last_hidden_state
y                     # 384 × seq_len × 2
```
"""
@concrete struct BertModel <: BertLikeModel
    embeddings
    encoder
    pooler
end
Flux.@layer BertModel

encoder_prefix(::BertModel) = ""

"""
    BertForSequenceClassification

transformers' `BertForSequenceClassification`, a [`BertModel`](@ref
Metalhead.Transformers.BertModel) with a linear layer on its pooled output, as the
cross-encoders for ranking and the classification models of BERT are, which
[`bert_for_sequence_classification`](@ref
Metalhead.Transformers.bert_for_sequence_classification) builds.

    (model::BertForSequenceClassification)(input_ids; attention_mask = nothing,
                                           token_type_ids = nothing)
    (model::BertForSequenceClassification)(inputs::NamedTuple)

The `logits` (num_labels, batch) of the token indices `input_ids` (seq_len, batch), or of
the output of the processor of the repository. A vector is one sequence, with logits
`(num_labels,)`. [`forward_features`](@ref) returns the hidden states of the base model.
"""
@concrete struct BertForSequenceClassification <: TextEncoderHead
    bert
    dropout
    classifier
end
Flux.@layer BertForSequenceClassification

encoder_model(m::BertForSequenceClassification) = m.bert
encoder_prefix(::BertForSequenceClassification) = "bert."

function forward_head(m::BertForSequenceClassification, x; pre_logits::Bool = false)
    pooled = m.dropout(forward_head(m.bert, x))
    return pre_logits ? pooled : m.classifier(pooled)
end

function (m::BertForSequenceClassification)(input_ids::AbstractMatrix{<:Integer};
        kwargs...)
    return forward_head(m, forward_features(m, input_ids; kwargs...))
end

"""
    TiedLinear(weight, bias)

An output projection `nn.Linear` whose weight is that of an `Embedding`: `weight` is its
`(hidden, vocab)` matrix itself, so the two stay tied, and `bias` a vector or `false`. Its key
`weight` is an alias, which a checkpoint may omit. It is `TiedLMHead` with a bias, as the
heads of the masked language models are.
"""
@concrete struct TiedLinear
    weight
    bias
end
Flux.@layer TiedLinear

function layer_params(l::TiedLinear)
    return [("weight", l.weight, :plain);
            l.bias isa AbstractArray ? [("bias", l.bias, :plain)] : []]
end

function (l::TiedLinear)(x)
    y = transpose(l.weight) * reshape(x, size(x, 1), :) .+ l.bias
    return reshape(y, :, size(x)[2:end]...)
end

# Builders

# The number of labels of a classification head, from `id2label` or `num_labels` of a config
# (transformers derives `num_labels` from the length of `id2label`, and defaults it to 2).
function num_labels_of(id2label, num_labels)
    if id2label !== nothing
        num_labels === nothing || num_labels == length(id2label) ||
            error("`num_labels = $num_labels` is incompatible with the `id2label` map of length $(length(id2label))")
        return length(id2label)
    end
    return something(num_labels, 2)
end

function check_encoder(name; is_decoder, add_cross_attention)
    is_decoder &&
        error("`is_decoder = true` is not supported by the Metalhead port of $name: only the encoder is")
    add_cross_attention &&
        error("`add_cross_attention = true` is not supported by the Metalhead port of $name: only the encoder is")
    return nothing
end

# A `BertModel` or `XLMRobertaModel` (the `model` type) of the layers of BERT, with the
# embeddings that `embeddings(norm, dropout)` builds, and a pooler if `add_pooling_layer`.
function bert_like(model, embeddings; hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob,
        initializer_range, layer_norm_eps, add_pooling_layer)
    hidden_size % num_attention_heads == 0 ||
        error("`hidden_size` ($hidden_size) must be divisible by `num_attention_heads` ($num_attention_heads)")
    act = act_fn(hidden_act; table = ENCODER_ACT2FN, key = "hidden_act")
    init = normal_init(initializer_range)
    dense(io) = init_layer(Dense, io; init)
    norm() = init_layer(LayerNorm, hidden_size; eps = Float32(layer_norm_eps))
    hidden_dropout() = dropout_or_identity(hidden_dropout_prob)
    function layer()
        attention = BertAttention(
            BertSelfAttention(num_attention_heads, dense(hidden_size => hidden_size),
                dense(hidden_size => hidden_size), dense(hidden_size => hidden_size),
                dropout_or_identity(attention_probs_dropout_prob)),
            BertSelfOutput(dense(hidden_size => hidden_size), norm(), hidden_dropout()))
        return BertLayer(attention,
            BertIntermediate(dense(hidden_size => intermediate_size), act),
            BertOutput(dense(intermediate_size => hidden_size), norm(), hidden_dropout()))
    end
    pooler = add_pooling_layer ? BertPooler(dense(hidden_size => hidden_size), tanh) :
             nothing
    return model(embeddings(norm, hidden_dropout),
        BertEncoder([layer() for _ in 1:num_hidden_layers]), pooler)
end

"""
    bert_model(; vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12,
        num_attention_heads = 12, intermediate_size = 3072, hidden_act = "gelu",
        hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 512, type_vocab_size = 2, initializer_range = 0.02,
        layer_norm_eps = 1e-12, add_pooling_layer = true)

Port of transformers' `BertModel` (see [`BertModel`](@ref Metalhead.Transformers.BertModel)
for its inputs and outputs), built from the fields of its `BertConfig` with the same names and
defaults. The decoder (`is_decoder`, `add_cross_attention`) is not supported. The activations
are `"gelu"` (the exact one), `"gelu_new"` and `"gelu_pytorch_tanh"` (the tanh
approximation), `"relu"`, `"silu"`, `"swish"` and `"quick_gelu"`.
`add_pooling_layer = false` builds a model without a pooler, for checkpoints that don't have
one (transformers initialises it at random for those).
"""
function bert_model(; vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12,
        num_attention_heads = 12, intermediate_size = 3072, hidden_act = "gelu",
        hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 512, type_vocab_size = 2, initializer_range = 0.02,
        layer_norm_eps = 1e-12, is_decoder = false, add_cross_attention = false,
        add_pooling_layer = true)
    check_encoder("BertModel"; is_decoder, add_cross_attention)
    init = normal_init(initializer_range)
    embedding(n) = init_layer(Embedding, n => hidden_size; init)
    function embeddings(norm, dropout)
        return BertEmbeddings(embedding(vocab_size), embedding(max_position_embeddings),
            embedding(type_vocab_size), norm(), dropout())
    end
    return bert_like(BertModel, embeddings; hidden_size, num_hidden_layers,
        num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob,
        attention_probs_dropout_prob, initializer_range, layer_norm_eps, add_pooling_layer)
end

"""
    bert_for_sequence_classification(; vocab_size = 30522, hidden_size = 768, ...,
        classifier_dropout = nothing, id2label = nothing, num_labels = nothing)

Port of transformers' `BertForSequenceClassification` (see [`BertForSequenceClassification`](@ref
Metalhead.Transformers.BertForSequenceClassification)), built from the fields of its
`BertConfig` (see [`bert_model`](@ref Metalhead.Transformers.bert_model)) with the same
names and defaults. The number of labels is the length of `id2label`, else `num_labels`,
else 2; `classifier_dropout` defaults to `hidden_dropout_prob`.
"""
function bert_for_sequence_classification(; vocab_size = 30522, hidden_size = 768,
        num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072,
        hidden_act = "gelu", hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 512, type_vocab_size = 2, initializer_range = 0.02,
        layer_norm_eps = 1e-12, is_decoder = false, add_cross_attention = false,
        classifier_dropout = nothing, id2label = nothing, num_labels = nothing)
    bert = bert_model(; vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob,
        max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps,
        is_decoder, add_cross_attention, add_pooling_layer = true)
    init = normal_init(initializer_range)
    return BertForSequenceClassification(bert,
        dropout_or_identity(something(classifier_dropout, hidden_dropout_prob)),
        init_layer(Dense, hidden_size => num_labels_of(id2label, num_labels); init))
end

# the position ids that old checkpoints store, which transformers no longer keeps
const POSITION_IDS_IGNORE = [r"embeddings\.position_ids$"]

register!(ArchSpec(TRANSFORMERS, "BertModel", bert_model; family = "bert",
    ignore = POSITION_IDS_IGNORE,
    references = [
        "sentence-transformers/all-MiniLM-L6-v2" => "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "BAAI/bge-small-en-v1.5" => "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"]))
register!(ArchSpec(TRANSFORMERS, "BertForSequenceClassification",
    bert_for_sequence_classification; family = "bert", ignore = POSITION_IDS_IGNORE,
    references = [
        "cross-encoder/ms-marco-MiniLM-L6-v2" => "233902d25c440f23af6f7d6e94d2946bac0bee0a"]))
MODEL_TYPE_CLASSES["bert"] = "BertModel"
