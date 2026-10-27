# Sentence transformers

See the [text embeddings tutorial](@ref embeddings) for using the models.

`create_model` builds a `SentenceTransformer` from a repository with a `modules.json`.
`encode` is HuggingFaceTokenizers' `encode`, to which `using HuggingFaceTokenizers` adds a
method for these models.

```@docs
Metalhead.SentenceTransformers.SentenceTransformer
Metalhead.SentenceTransformers.encode
Metalhead.SentenceTransformers.encode_query
Metalhead.SentenceTransformers.encode_document
Metalhead.SentenceTransformers.similarity
Metalhead.SentenceTransformers.preprocess
Metalhead.SentenceTransformers.Transformer
Metalhead.SentenceTransformers.Pooling
Metalhead.SentenceTransformers.Dense
Metalhead.SentenceTransformers.Normalize
```

