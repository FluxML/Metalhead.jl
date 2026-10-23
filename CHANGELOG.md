# Changelog

All notable changes to Metalhead.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases before 0.9.6 are described
in the [GitHub releases](https://github.com/FluxML/Metalhead.jl/releases).

## [Unreleased] (0.10.0)

### Added

- Flux ports of [timm](https://github.com/huggingface/pytorch-image-models) models that load
  their weights directly from the Hugging Face Hub (#305, #307, #319, #320):
  - `create_model` builds a model from a Hub repository (`"timm/resnet50.a1_in1k"`), a bare
    architecture (`"resnet50"`) or a local directory. Weights are random unless
    `pretrained = true`. It takes timm's keyword arguments (`num_classes`, `in_chans`, ...),
    plus `pretrained`, `revision` and `eltype`.
  - `list_models`, `forward_features`, `forward_head` and `Metalhead.Timm.pretrained_cfg`,
    which returns the preprocessing settings of a checkpoint.
  - Architecture names are unique across libraries. A library registers its architectures with
    a `Metalhead.ConfigFormat`, which says how its `config.json` names an architecture; timm's
    is in `src/Timm/config.jl`.
  - `forward_intermediates` and `feature_info`, for the multi-scale features of segmentation
    and depth heads (ResNet and its variants, ConvNeXt, ViT, the MLP mixers, Inception v4).
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
- Validation against PyTorch: float64 parity fixtures for every code path, run in CI without
  Python or network access, and float32 references for 56 pretrained checkpoints. Weights load
  strictly by name.
- `porting/` tools and a Claude Code skill (`skills/port-hf-model`) for porting new
  models (#305, #307).
- Documentation: getting started, model list, validation, porting and a
  [migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/) from 0.9 (#307).

### Changed

- New dependencies: HuggingFaceApi, SafeTensors, JSON and ConcreteStructs. Flux 0.16.13 and
  NNlib 0.9.51 are now the minimum versions (#305).
- Downloads share the cache of Python's `huggingface_hub` (`~/.cache/huggingface/hub`), and
  its `HF_TOKEN`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` environment variables.

### Deprecated

- The hand-written models (`ResNet(50)`, `ViT(:base)`, ...) and their lowercase builders, the
  `Metalhead.Layers` module, and `pretrain = true` with the weights it loads from the FluxML
  organisation. They will be removed in 0.11 (#305).

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
