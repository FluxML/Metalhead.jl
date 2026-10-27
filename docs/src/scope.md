# [Scope](@id scope)

Metalhead ports what a checkpoint determines, and checks it against PyTorch. It doesn't
reimplement the general-purpose machinery of timm, transformers, diffusers and
sentence-transformers. A feature belongs in Metalhead when both of these hold:

1. The checkpoint determines it: through its configuration, its weights, or upstream code
   specific to its architecture.
2. Getting it wrong would silently change the results compared with the Python library, so
   it has to be validated against them.

In scope:

- Architectures, their configurations and weights, and the feature interface of image and
  language models (`forward_features`, `forward_intermediates`, ...).
- The inference preprocessing that a checkpoint's configuration sets: resizing, cropping,
  normalising, and the packing of inputs: the token indices and padding of a batch of texts,
  and what is specific to a model, such as image patches and image placeholder tokens.
- The post-processing specific to a model, such as SAM 3's masks and boxes.
- The schedulers and sampling loops of the ported diffusion pipelines. The sampler, not the
  denoiser alone, defines the image. A scheduler is ported along with a pipeline that uses
  it, not on its own.
- The text embeddings of sentence-transformers models. sentence-transformers is a library of
  its own, but a checkpoint sets how its embeddings are made, and getting it wrong changes
  them silently: the pipeline of its `modules.json` (the pooling of the token embeddings,
  its mode, `Dense` projections and normalization), the truncation and lowercasing of
  `sentence_bert_config.json`, and the prompts and similarity function of
  `config_sentence_transformers.json`. Metalhead reads them strictly and computes the
  embeddings as sentence-transformers' `encode` does.
- The key-value cache of language models (not implemented yet).

Out of scope, for other packages:

- Tokenization algorithms. HuggingFaceTokenizers.jl tokenizes, and Metalhead uses it through
  an extension.
- Chat templates, and decoding strategies such as beam search, logits processors and
  streaming.
- Training: optimizers, learning-rate schedules, losses, weight averaging and training loops,
  which Flux, Optimisers.jl and ParameterSchedulers.jl provide. The models are ordinary
  differentiable Flux models.
- Data augmentation, data loading and image I/O, which DataAugmentation.jl, MLUtils.jl and
  FileIO.jl provide.
- Task APIs such as transformers' `pipeline`, quantization, LoRA and distributed training.
- The rest of sentence-transformers: training and its losses, evaluators, semantic search and
  similarity utilities other than the similarity function a model sets, quantized
  embeddings, and multi-process encoding.

General-purpose code that matching PyTorch requires, such as PIL's resampling filters, stays
internal rather than public. Architectures are added once a checkpoint validates them; the
goal is not to cover each library's whole model collection.
