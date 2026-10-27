# BERT, XLM-RoBERTa and ModernBERT

The text encoders of transformers: [`BertModel`](@ref Metalhead.Transformers.BertModel)
(MiniLM, the BGE and GTE embedding models, ...),
[`XLMRobertaModel`](@ref Metalhead.Transformers.XLMRobertaModel) (the multilingual E5 models,
BGE-M3 and the BGE rerankers' backbone, ...) and
[`ModernBertModel`](@ref Metalhead.Transformers.ModernBertModel) (ModernBERT and the models
built on it). They map the token indices of their processor, the
[`TextProcessor`](@ref Metalhead.TextProcessor) of the repository, to the hidden states of
the tokens; a head on top gives the logits of a classifier, such as a cross-encoder that ranks
passages, or of a masked language model.

```julia
using Metalhead, HuggingFaceTokenizers

model = create_model("answerdotai/ModernBERT-base")   # a ModernBertForMaskedLM
proc = create_processor("answerdotai/ModernBERT-base")
inputs = proc(["The capital of France is [MASK].", "Dogs bark."])
hidden = forward_features(model, inputs)    # 768 × seq_len × 2: its ModernBertModel's output
logits = model(inputs)                      # 50368 × seq_len × 2

reranker = create_model("BAAI/bge-reranker-base")
proc = create_processor("BAAI/bge-reranker-base")
inputs = proc(["what is a panda?", "what is a panda?"];
    text_pair = ["The giant panda is a bear native to China.", "Paris is the capital of France."])
reranker(inputs)                            # 1 × 2 scores, the first higher
```

The models take `input_ids` and the `attention_mask` that excludes the padding of a batch
(and BERT's `token_type_ids`), or the output of the processor as it is. The processors of
these repositories return the keys of transformers' tokenizers: `token_type_ids` too for
BERT's, and with `text_pair` the two texts of a cross-encoder joined by the tokenizer's
template. Pooling the hidden states into one embedding per text, as sentence-transformers
does, is not part of these models. The repositories of embedding models (`all-MiniLM-L6-v2`,
`bge-small-en-v1.5`, `multilingual-e5-base`, `gte-modernbert-base`, ...) have a modules.json:
[`create_model`](@ref) builds the sentence-transformers model over them (see
[Sentence transformers](sentence_transformers.md)), and `create_processor` has no processor
for it. Their `BertModel`, `XLMRobertaModel` or `ModernBertModel` is the one that the
`config.json` names, which the references of these repositories validate.

## Models

```@docs
Metalhead.Transformers.TextEncoder
Metalhead.Transformers.BertModel
Metalhead.Transformers.BertForSequenceClassification
Metalhead.Transformers.XLMRobertaModel
Metalhead.Transformers.XLMRobertaForSequenceClassification
Metalhead.Transformers.ModernBertModel
Metalhead.Transformers.ModernBertForMaskedLM
Metalhead.Transformers.ModernBertForSequenceClassification
```

## Builders

`create_model` calls these with the fields of the repository's `config.json`.

```@docs
Metalhead.Transformers.bert_model
Metalhead.Transformers.bert_for_sequence_classification
Metalhead.Transformers.xlm_roberta_model
Metalhead.Transformers.xlm_roberta_for_sequence_classification
Metalhead.Transformers.modernbert_model
Metalhead.Transformers.modernbert_for_masked_lm
Metalhead.Transformers.modernbert_for_sequence_classification
```
