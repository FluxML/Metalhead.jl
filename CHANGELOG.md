# Changelog

All notable changes to Metalhead.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases before 0.9.6 are described
in the [GitHub releases](https://github.com/FluxML/Metalhead.jl/releases).

## [Unreleased] (0.10.0)

### Added

- `list_models(; byfamily = true)` returns a dictionary of architectures by family. Combine
  with `pretrained = true` to group validated checkpoints. The documentation uses the same
  registry metadata.
- Flux ports of [timm](https://github.com/huggingface/pytorch-image-models) models that load
  their weights directly from the Hugging Face Hub (#305, #307, #319, #320):
  - `create_model` builds a model from a Hub repository (`"timm/resnet50.a1_in1k"`), a bare
    architecture (`"resnet50"`) or a local directory. Weights are random unless
    `pretrained = true`. It takes timm's keyword arguments (`num_classes`, `in_chans`, ...),
    plus `pretrained`, `revision` and `eltype` (`Float16`, `Flux.BFloat16`, `Float32` or
    `Float64`).
  - `list_models`, `forward_features`, `forward_head` and `Metalhead.Timm.pretrained_cfg`,
    which returns the preprocessing settings of a checkpoint.
  - Architecture names are unique across libraries. A library registers its architectures with
    a `Metalhead.ConfigFormat`, which says how its `config.json` names an architecture; timm's
    is in `src/Timm/config.jl`.
  - `forward_intermediates` and `feature_info`, for the multi-scale features of segmentation
    and depth heads (ResNet and its variants, ConvNeXt, ViT, the MLP mixers, DenseNet, VGG,
    Inception v3/v4, Inception-ResNet-v2, aligned Xception, and the EfficientNet and MobileNet
    families, including MixNet, MNASNet, FBNet and LCNet) (#309).
  - ViT options of DINOv2, CLIP and SigLIP: layer scale, register tokens, pre-norm,
    QuickGELU, attention pooling (`global_pool = "map"`), average/max pooling, and
    `dynamic_img_size`, which resamples the position embedding like PyTorch's antialiased
    bicubic interpolation.
  - 298 architectures: ResNet, ResNeXt, Wide ResNet, SE-ResNet, SE-ResNeXt, SENet, ECA-ResNet,
    Res2Net and Res2NeXt; ViT, including DINOv2 (with registers), CLIP and SigLIP; ConvNeXt;
    VGG; DenseNet; Inception v3 and v4, Inception-ResNet-v2; aligned Xception; MLP-Mixer, ResMLP and gMLP; ConvMixer;
    EfficientNet (B0–B8, L2, Lite, EdgeTPU, CondConv) and EfficientNetV2; MobileNet v1, v2,
    v3 and v4; MobileNet-EdgeTPU; MNASNet, SemNASNet, SPNASNet and TinyNet; MixNet; FBNet-C
    and FBNetV3; LCNet.
- Flux ports of [transformers](https://github.com/huggingface/transformers) language models,
  `LlamaForCausalLM` (#327), `Qwen2ForCausalLM` and `Qwen3ForCausalLM` (#329), and
  `Gemma3ForCausalLM`, the Gemma 3 text model, with its sliding-window attention, which
  NNlib computes over the keys of the windows only (#336), and
  IBM's Granite (`GraniteForCausalLM`), its mixtures of experts (`GraniteMoeForCausalLM` and
  `GraniteMoeSharedForCausalLM`) and Granite 4.0 without its Mamba layers
  (`GraniteMoeHybridForCausalLM`, #337), validated against `HuggingFaceTB/SmolLM2-135M`,
  `unsloth/Llama-3.2-1B`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen3-0.6B`, `unsloth/gemma-3-270m`,
  `ibm-granite/granite-3.3-2b-base`, `ibm-granite/granite-3.1-1b-a400m-base`,
  `ibm-research/moe-7b-1b-active-shared-experts` and `ibm-granite/granite-4.0-350m-base`.
  The rotary embeddings may be scaled as in Llama 3.1 and later (`rope_type = "llama3"`) or
  linearly (`"linear"`) (#336). `create_model` loads them from their Hub repository, and
  takes the fields of `config.json` as keyword arguments. They take token indices starting
  at 1 and return logits; there is no key-value cache yet. Batches padded on the right take
  the tokenizer's `attention_mask`, and `forward_intermediates` returns the hidden states of
  the decoder layers.
- Flux ports of [diffusers](https://github.com/huggingface/diffusers) models and pipelines, in
  `Metalhead.Diffusers`: text-to-image generation with FLUX.2 [klein] (`Flux2KleinPipeline`,
  with `Flux2Transformer2DModel`, `AutoencoderKLFlux2` and `FlowMatchEulerDiscreteScheduler`),
  validated against `black-forest-labs/FLUX.2-klein-4B` (#332). `create_model` builds a pipeline from
  the `model_index.json` of a repository, or one of its components with `subfolder`
  (`"org/name/vae"` also names one). With HuggingFaceTokenizers.jl loaded, pipelines take
  prompts as strings.
- Mixed precision under `eltype = Float16` or `Flux.BFloat16` (or `f16`/`bf16`, #332): the
  RMSNorms of the language models and timm's `LayerNorm2d` keep their parameters in
  `Float32`, as Flux's `BatchNorm`, `GroupNorm` and `LayerNorm` do, and normalise in
  `Float32`, while the activations stay in half precision. ViTs resample their position
  embeddings in `Float32`, as timm does.
- Validation against PyTorch: float64 parity fixtures for every code path, run in CI without
  Python or network access, and float32 references for 59 pretrained checkpoints. Weights load
  strictly by name.
- `porting/` tools and an agent skill (`skills/port-hf-model`) for porting new
  models (#305, #307).
- Documentation: getting started, a model list grouped by family, validation, porting and a
  [migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/) from 0.9 (#307).
- A tutorial on vision models covering image preprocessing, classification, transfer learning,
  multi-scale features and vision transformer tokens.

### Changed

- New dependencies: HuggingFaceApi, SafeTensors, JSON and ConcreteStructs. Flux 0.16.14,
  NNlib 0.9.54 and MLUtils 0.4.9 are now the minimum versions (#305, #327, #337).
- Downloads share the cache of Python's `huggingface_hub` (`~/.cache/huggingface/hub`), and
  its `HF_TOKEN`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` environment variables.

### Deprecated

- The hand-written models (`ResNet(50)`, `ViT(:base)`, ...) and their lowercase builders, the
  `Metalhead.Layers` module, and `pretrain = true` with the weights it loads from the FluxML
  organisation. They will be removed in 0.11 (#305).

### Removed

- Deprecated API reference pages and the old ResNet tutorial from the documentation.

### Fixed

- `MobileNetv3` on Julia 1.13 (#303).

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

[Unreleased]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.7...HEAD
[0.9.7]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.5...v0.9.6
