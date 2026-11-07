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
- The generation of language models: `generate`, transformers' greedy decoding and
  multinomial sampling, with the settings of the checkpoint's `generation_config.json` (the
  temperature, top-k, top-p and min-p of sampling, the repetition penalty, the minimum length
  and the end-of-text tokens), which change the text silently if they are wrong; and the
  key-value cache, which `generate` uses and which serves the user's own loop too. The cache
  is implemented for the models whose layers are all full attention (Llama, Qwen2, Qwen3,
  Granite and their mixtures of experts), not yet for Gemma 3 and Qwen3.5, whose `generate`
  runs the model on the whole sequence at each step.

Out of scope, for other packages:

- Tokenization algorithms. HuggingFaceTokenizers.jl tokenizes, and Metalhead uses it through
  an extension.
- Chat templates, and the other decoding strategies and logits processors of transformers,
  such as beam search, contrastive search, assisted decoding, `no_repeat_ngram_size`,
  `suppress_tokens`, stop strings and streaming. A `generation_config.json` that sets one is
  an error.
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
