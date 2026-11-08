# Changelog

All notable changes to Metalhead.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases before 0.9.6 are described
in the [GitHub releases](https://github.com/FluxML/Metalhead.jl/releases).

## [0.10.0]

Breaking: Flux 0.16.14, NNlib 0.9.57 and MLUtils 0.4.9 are now the minimum versions, and the
hand-written models of 0.9 are deprecated, with warnings (see "Deprecated" and the
[migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/)).

### Added

Ports of Hugging Face Hub models, loaded by name from their original safetensors weights and
checked numerically against PyTorch (float64 fixtures run in CI without Python or network, and
103 pretrained checkpoints validated against float32 references):

- [timm](https://github.com/huggingface/pytorch-image-models) vision models, 298
  architectures: ResNet and its variants (ResNeXt, SE-ResNet, ECA-ResNet, Res2Net, ...), ViT
  (with DINOv2, CLIP and SigLIP options), ConvNeXt, VGG, DenseNet, Inception, aligned
  Xception, MLP-Mixer, ResMLP, gMLP, ConvMixer, EfficientNet (V1, V2, Lite, EdgeTPU,
  CondConv), MobileNet v1 to v4, MNASNet, MixNet, FBNet and LCNet (#305, #307, #319, #320,
  #321).
  - `create_model` builds a model from a Hub repository (`"timm/resnet50.a1_in1k"`), a bare
    architecture (`"resnet50"`) or a local directory, with timm's keyword arguments
    (`num_classes`, `in_chans`, ...) and `pretrained`, `revision` and `eltype` (`Float16`,
    `Flux.BFloat16`, `Float32` or `Float64`). A bare architecture has random weights unless
    `pretrained = true`, which loads the checkpoint of
    `timm.create_model(name, pretrained=True)` (#391). The weights of 0.9 for ResNet-50 and
    Wide ResNet-50-2 are `timm/resnet50.tv2_in1k` and `timm/wide_resnet50_2.tv2_in1k`.
  - `list_models` (`byfamily = true` groups by family, `pretrained = true` lists the
    validated checkpoints), `forward_features`, `forward_head`,
    `Metalhead.Timm.pretrained_cfg` (the preprocessing settings of a checkpoint),
    `forward_intermediates` and `feature_info` (the multi-scale features of segmentation and
    depth heads) (#321, #325).
- [transformers](https://github.com/huggingface/transformers) language models: Llama,
  Qwen2, Qwen3, Gemma 3 (with its sliding-window attention), IBM Granite (with its mixtures
  of experts and Granite 4.0 without its Mamba layers), Qwen3 MoE, and the vision-language
  Qwen3.5, Qwen3.5 MoE and Qwen3.8 (gated delta rule layers, multimodal rotary embedding;
  no videos) (#327, #329, #336, #337, #340, #371). They take token indices from 1 and
  return logits, a batch padded on the right with its `attention_mask`, or the output of
  their processor (`model(tok(texts))`). `forward_intermediates` returns the hidden states of
  the decoder layers. The checkpoints that transformers 5 saves for Qwen3.5 MoE, with one MLP
  per expert, don't load.
  - `Metalhead.Transformers.KVCache`, for the models whose layers are all full attention
    (Llama, Qwen2, Qwen3, Qwen3 MoE, Granite): `model(input_ids; cache)` returns
    `(logits, cache)` (#390). Gemma 3 and Qwen3.5 raise a "not supported yet" error.
  - `generate(model, input_ids)`: greedy decoding and sampling (temperature, top-k, top-p,
    min-p), with the repetition penalty, the minimum length, the end-of-text tokens and the
    settings of the checkpoint's `generation_config.json` (the `generation_config` field of
    the model, overridden by keyword arguments), validated against transformers' (#392).
- The text encoders BERT, XLM-RoBERTa and ModernBERT, with their sequence classification
  heads (and ModernBERT's masked language model), the backbones of embedding models, rerankers
  and masked language models (#378).
- The dual encoders CLIP (`CLIPModel`, with `CLIPTextModel` and `CLIPVisionModel`) and
  SigLIP and SigLIP 2 (`SiglipModel`, `Siglip2Model`, with the NaFlex checkpoints), for
  zero-shot classification and image-text retrieval, with `get_text_features` and
  `get_image_features` and their processors (#368, #375), and a tutorial.
- SAM 3 (`Sam3Model`), which finds and segments the objects that a text or box prompt
  describes, validated against SAM 3 and SAM 3.1. The checkpoints load as `Sam3VideoModel`,
  of which only the detector is ported (#343).
- [diffusers](https://github.com/huggingface/diffusers) pipelines, in `Metalhead.Diffusers`:
  text-to-image generation with FLUX.2 [klein] (`Flux2KleinPipeline`, #332), with image
  editing from reference images (#365), and with Z-Image-Turbo (`ZImagePipeline`, #369).
  Both take `num_images_per_prompt` (#396). `create_model` builds a pipeline from the
  `model_index.json` of a repository, or one of its components with `subfolder`.
  `Metalhead.Diffusers.encode_prompt(pipe, prompts)` embeds prompts as diffusers does.
- [sentence-transformers](https://github.com/huggingface/sentence-transformers) text
  embeddings, in `Metalhead.SentenceTransformers`, validated against
  `Qwen/Qwen3-Embedding-0.6B`: `create_model` builds a `SentenceTransformer` from a repository
  with a `modules.json` (`Transformer`, `Pooling`, `Dense` and `Normalize` modules, prompts,
  similarity function and `truncate_dim`), with `encode`, `encode_query`, `encode_document`
  and `similarity` (#374).

Around the models:

- Processors (#348, #352-#358): `create_processor` loads the processor of a checkpoint, which
  turns raw inputs (images, texts, boxes) into the inputs of the model and its outputs back
  into results, as `AutoProcessor` does: timm's eval transform (`Metalhead.Timm.EvalTransform`,
  bit for bit on 8-bit images), and those of CLIP, SigLIP, Qwen3.5 and SAM 3 (with the
  `post_process_*` functions). Settings are read strictly from the repository's files and
  overridden by keyword arguments; a setting that the port doesn't implement is an error.
- An extension for HuggingFaceTokenizers.jl: it loads the `Tokenizer` of a checkpoint, so
  that processors and pipelines take strings, and `create_processor` of a language model
  returns a `Metalhead.TextProcessor` (Metalhead ports no tokenizer). An extension for
  ImageCore lets processors take matrices of colorants and gives `Metalhead.image_matrix`.
- `create_model` warns about the keys of a transformers or diffusers `config.json` that the
  port doesn't read, instead of dropping them silently (#367), and merges the sub-configs in
  its keywords (`text_config = (; num_hidden_layers = 4)`) into those of the repository
  (#396).
- Mixed precision under `eltype = Float16` or `Flux.BFloat16` (or `f16`/`bf16`, #332): the
  RMSNorms of the language models and timm's `LayerNorm2d` keep their parameters in `Float32`
  and normalise in `Float32`, as Flux's norm layers do, while the activations stay in half
  precision. ViTs resample their position embeddings in `Float32`, as timm does, and the
  `A_log` of the Qwen3.5 linear attention stays `Float32`, as in its checkpoints. Pretrained
  models allocate their weights directly in the requested `eltype` (#366).
- The public API is declared: what you touch after `create_model` is exported or, on Julia
  1.11 and later, `public`. The builders, the fields of model structs and the porting API
  (`ArchSpec`, `register!`, `AbstractConfigFormat`, `load_state!`) stay internal and are
  documented on a page of the API reference (#394).
- `porting/` tools and an agent skill (`skills/port-hf-model`) for porting new models, and
  documentation: getting started, a model list grouped by family, validation, a gallery of
  the outputs of the models, tutorials (vision, vision-language, segmentation, language
  models, text-to-image, zero-shot classification, embeddings) and a
  [migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/) from 0.9 (#307, #361).

### Changed

- New dependencies: HuggingFaceApi, SafeTensors, JSON and ConcreteStructs, and the weak
  dependencies HuggingFaceTokenizers and ImageCore (#335, #337, #341, #388, #389).
- Downloads share the cache of Python's `huggingface_hub` (`~/.cache/huggingface/hub`), and
  its `HF_TOKEN`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` environment variables.
- The fixtures and reference outputs of the tests are not in the package (2.6 MB): they are
  lazy artifacts on the Hub dataset `FluxML/metalhead-test-data`, downloaded on first use
  (#372).

### Deprecated

- The hand-written models (`ResNet(50)`, `ViT(:base)`, ...) and their lowercase builders,
  `backbone`, `classifier` and `Metalhead.loadpretrain!`, the `Metalhead.Layers` module, and
  `pretrain = true` with the weights it loads from the FluxML organisation. They will be
  removed in 0.11 (#305). The model constructors, `backbone`, `classifier` and
  `loadpretrain!` warn once per session, naming their replacement (#345).

### Removed

- Deprecated API reference pages and the old ResNet tutorial from the documentation.

### Fixed

- `MobileNetv3` on Julia 1.13 (#303).
- On CUDA GPUs, the mean pooling of Inception v3 and v4 and Res2Net on feature maps of 1 × 1
  or 2 × 2, wrong with cuDNN before NNlib 0.9.55.

## [0.9.8] - 2026-10-28

### Fixed

- `SqueezeNet(; pretrain = true)` failed with `invalid struct allocation` on current Flux: the
  pretrained weights load as plain arrays instead of an outdated serialized Flux model (#384).

## [0.9.7] - 2026-10-23

### Changed

- Julia 1.10 and Flux 0.16 are now the minimum versions (#298).
- Compat with CUDA 6 (#294, #295).

### Removed

- The PartialFunctions dependency (#298).

### Fixed

- Loading pretrained ResNets (#287) and moving models with `Flux.gpu` (#286) (#290, #298).
- `show` of `LayerNormV2` (#298).

## [0.9.6] - 2026-01-05

### Changed

- Compat with JLD2 0.6 (#291).

[0.10.0]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.8...HEAD
[0.9.8]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.7...v0.9.8
[0.9.7]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.5...v0.9.6
