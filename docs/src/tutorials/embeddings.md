# [Text embeddings](@id embeddings)

An embedding model maps a text to a vector, so that texts of similar meaning get nearby
vectors: the basis of semantic search, retrieval-augmented generation, clustering and
deduplication. Most of the embedding models of the Hub are
[sentence-transformers](https://github.com/huggingface/sentence-transformers) models: a
transformers model, whose token embeddings a `Pooling` module turns into one embedding per
text, which `Dense` modules may project and `Normalize` normalizes. Their repository lists
these modules in a `modules.json`, and `create_model` builds them all:

```julia
using Metalhead, HuggingFaceTokenizers

model = create_model("Qwen/Qwen3-Embedding-0.6B")
```

```
SentenceTransformer
  Transformer: LlamaModel, max_seq_length = 32768
  Pooling: lasttoken
  Normalize
  prompts: query, document
```

The model is a [`SentenceTransformer`](@ref Metalhead.SentenceTransformers.SentenceTransformer)
whose first module holds the ported transformers model (`model[1].auto_model`, the decoder of
a Qwen3 language model here) and its tokenizer, a `Tokenizer` of
[HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl), which
encoding texts needs. The settings of the repository (the pooling, the prompts, the maximum
length of the texts) are read from its configuration files, strictly: a setting that
Metalhead doesn't implement is an error, and an unknown one a warning.

## Encoding texts

`encode` (HuggingFaceTokenizers' `encode`, which takes a model) returns the embeddings of
texts, one per column:

```julia
embeddings = encode(model, ["The weather is lovely today.", "It's so sunny outside!",
    "He drove to the stadium."])
size(embeddings)                          # (1024, 3)
similarity(model, embeddings, embeddings) # 3 × 3, cosine similarities
```

[`similarity`](@ref) compares embeddings with the similarity function of the model
(`model.similarity_fn_name`, the cosine for most), as sentence-transformers' `similarity`.
Texts are encoded in batches of `batch_size` (32 by default) on the device of the model, which
`gpu(model)` moves to the GPU, and are truncated to `model[1].max_seq_length` tokens.

## Queries and documents

Retrieval models such as Qwen3-Embedding encode queries with an instruction, a prompt of the
model, and documents without one. [`encode_query`](@ref) and [`encode_document`](@ref) use the
prompts of the model named `"query"` and `"document"`, as sentence-transformers' methods of
the same names do:

```julia
queries = encode_query(model, ["What is the capital of China?", "Explain gravity"])
documents = encode_document(model, ["The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to " *
    "physical objects and is responsible for the movement of planets around the sun."])
similarity(model, queries, documents)
```

```
2×2 Matrix{Float32}:
 0.764557  0.141425
 0.135498  0.599956
```

The prompts are in `model.prompts`; `encode` takes another one by name (`prompt_name`) or
as a string (`prompt`). Models trained to work with fewer features (Matryoshka embeddings)
take `truncate_dim`, which `create_model(repo; truncate_dim = 256)` sets for every call.

## Inputs and outputs

- `Metalhead.SentenceTransformers.preprocess(model, texts; prompt)` returns the inputs of the
  model: the token indices `input_ids` (from 1) and the `attention_mask`, of size
  `(seq_len, batch)`, padded on the right.
- `model(inputs)` applies the modules in order and returns the inputs with their outputs, the
  `token_embeddings` of size `(hidden_size, seq_len, batch)` and the `sentence_embedding` of
  size `(dim, batch)`, as sentence-transformers' `forward`. It is an ordinary Flux model,
  which can be fine-tuned.
- `encode` returns a matrix of size `(dim, length(texts))` on the CPU, or a vector for a single
  string.
- The weights are loaded in `Float32` by default (`eltype` of `create_model`), where
  sentence-transformers loads those of Qwen3-Embedding in `bfloat16`, their type on the Hub.
