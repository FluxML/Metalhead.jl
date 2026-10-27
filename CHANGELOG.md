# Changelog

All notable changes to Metalhead.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases before 0.9.6 are described
in the [GitHub releases](https://github.com/FluxML/Metalhead.jl/releases).

## [Unreleased]

### Fixed

- Pretrained models allocate their weights directly in the requested `eltype`, without
  random initialization or a temporary Float32 model. Mixed precision normalization and
  random initialization of replacement classifiers are preserved (#366).

## [0.10.0]

Breaking: Flux 0.16.14, NNlib 0.9.54 and MLUtils 0.4.9 are now the minimum versions, and the
hand-written models of 0.9 are deprecated, with warnings (see "Deprecated" and the
[migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/)).

### Added

- `list_models(; byfamily = true)` returns a dictionary of architectures by family. Combine
  with `pretrained = true` to group validated checkpoints. The documentation uses the same
  registry metadata.
- Flux ports of [timm](https://github.com/huggingface/pytorch-image-models) models that load
  their weights directly from the Hugging Face Hub (#305, #307, #319, #320):
  - `create_model` builds a model from a Hub repository (`"timm/resnet50.a1_in1k"`), a bare
    architecture (`"resnet50"`) or a local directory. A repository or a directory loads its
    weights; a bare architecture has random weights unless `pretrained = true`, which loads
    those of its default validated checkpoint (`pretrained = false` builds a repository's
    architecture with random weights). It takes timm's keyword arguments (`num_classes`,
    `in_chans`, ...), plus `pretrained`, `revision` and `eltype` (`Float16`,
    `Flux.BFloat16`, `Float32` or `Float64`).
  - `list_models`, `forward_features`, `forward_head` and `Metalhead.Timm.pretrained_cfg`,
    which returns the preprocessing settings of a checkpoint.
  - Architecture names are unique across libraries. A library registers its architectures with
    a `Metalhead.AbstractConfigFormat`, which says how its `config.json` names an
    architecture; timm's is in `src/Timm/config.jl`.
  - `forward_intermediates` and `feature_info`, for the multi-scale features of segmentation
    and depth heads (ResNet and its variants, ConvNeXt, ViT, the MLP mixers, DenseNet, VGG,
    Inception v3/v4, Inception-ResNet-v2, aligned Xception, and the EfficientNet and MobileNet
    families, including MixNet, MNASNet, FBNet and LCNet) (#321).
  - ViT options of DINOv2, CLIP and SigLIP: layer scale, register tokens, pre-norm,
    QuickGELU, attention pooling (`global_pool = "map"`), average/max pooling, and
    `dynamic_img_size`, which resamples the position embedding like PyTorch's antialiased
    bicubic interpolation (#321).
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
  the tokenizer's `attention_mask`, and the models take the output of their processor as it
  is (`model(tok(texts))`). `forward_intermediates` returns the hidden states of the decoder
  layers, for a batch or a single sequence, and `feature_info` gives their modules.
- Qwen3.5, the vision-language model of the Qwen3.5 checkpoints
  (`Qwen3_5ForConditionalGeneration`), validated against `Qwen/Qwen3.5-0.8B` on text and on
  an image (#340). Its decoder alternates gated attention with gated delta rule (linear
  attention) layers and has a multimodal rotary embedding; it is also ported alone
  (`Qwen3_5ForCausalLM`), as is its vision encoder (`Qwen3_5VisionModel`). It takes the
  output of its processor (#348). Videos are not supported.
- SAM 3, the detector of the Segment Anything Model 3 (`Sam3Model`), which finds and segments
  the objects of an image that a text prompt or box prompts describe, validated against
  `jetjodh/sam3`, an ungated copy of the weights of `facebook/sam3`, and SAM 3.1
  (`unsloth/sam3.1`, #343). Its ViT attends within windows with a 2D rotary embedding, and its
  CLIP text encoder, geometry encoder of boxes, DETR encoder and decoder and mask decoder
  are ported. The checkpoints of SAM 3 load as `Sam3VideoModel`, whose detector only is
  ported, not its tracker.
- Flux ports of [diffusers](https://github.com/huggingface/diffusers) models and pipelines, in
  `Metalhead.Diffusers`: text-to-image generation with FLUX.2 [klein] (`Flux2KleinPipeline`,
  with `Flux2Transformer2DModel`, `AutoencoderKLFlux2` and `FlowMatchEulerDiscreteScheduler`),
  validated against `black-forest-labs/FLUX.2-klein-4B` (#332). `create_model` builds a
  pipeline from the `model_index.json` of a repository, or one of its components with
  `subfolder` (`"org/name/vae"` also names one). The tokenizer of a pipeline,
  `pipe.tokenizer`, is HuggingFaceTokenizers' `Tokenizer` of its `tokenizer/` subfolder,
  which `create_model` loads with the extension (a pipeline loaded before the extension
  loads it once the extension is loaded), and
  `Metalhead.Diffusers.encode_prompt(pipe, prompts)` embeds prompts as diffusers'
  `encode_prompt` does, with the chat template of the text encoder (#348).
- An extension for HuggingFaceTokenizers.jl (`ext/`): with it loaded, `create_processor`
  returns the processor of a language model, a `Metalhead.TextProcessor`, which encodes
  texts into the 1-based token indices and the attention mask of the models, padded on the
  right, and decodes token indices with `decode(tok, indices)`. Its tokenizer is
  HuggingFaceTokenizers' `Tokenizer` of the repository's `tokenizer.json` with the padding
  token of its `tokenizer_config.json` (Metalhead ports no tokenizer, #348), and processors
  and pipelines take text as strings (#332, #348).
- Processors (#348): `create_processor` loads the processor of a checkpoint, which turns raw
  inputs (images, text, boxes) into the inputs of the model, and its outputs back into
  results, as the checkpoint's configuration and its Python library do. It takes the names
  that `create_model` takes, at the same revisions, reads the files of a branch at the commit
  of the first it finds (offline or without network, those of the cache), and returns the
  most complete processor that the repository describes, as transformers' `AutoProcessor`
  does:
  - `Metalhead.Timm.EvalTransform`, timm's eval transform, for every timm checkpoint, with
    `use_test_size`. It resizes with a port of Pillow's 8-bit filters, matches timm bit for
    bit on 8-bit images, and returns an array.
  - `Metalhead.Transformers.Qwen3VLProcessor` and its `Qwen2VLImageProcessor`, for Qwen3.5,
    with the tokenizer of the repository for its text. It expands each image placeholder of
    a prompt into the image's tokens.
  - `Metalhead.Transformers.Sam3Processor` and its `Sam3ImageProcessor`, for SAM 3, with box
    prompts, and the tokenizer of the repository (HuggingFaceTokenizers' `Tokenizer`) for its
    text. `post_process_instance_segmentation`, `post_process_object_detection` and
    `post_process_semantic_segmentation`, exported by Metalhead, take the processor and the
    outputs of the model. Image sizes, such as the `original_sizes` of its output and the
    `target_sizes` of post-processing, are `(; height, width)`; `(height, width)` tuples
    are accepted too.

  Settings are read from the repository's files over the defaults of the Python class,
  strictly: a setting that the port doesn't implement is an error, and an unknown key a
  warning. Keyword arguments override them, also in natural forms (an integer `input_size`
  or `size`, one `mean` for every channel, a Symbol `interpolation` or `resample`), and
  `show` prints where each comes from. The processor classes are ported along with their
  models, and a class that isn't ported is an error. A class is a struct defined with
  `Metalhead.@processor` and registered with `Metalhead.register_processor!`, whose
  constructors check its settings with `Metalhead.standardize_settings` (see "Porting
  internals" in the API reference). Processors other than timm's return NamedTuples with
  the keys of transformers, in Julia's layout, with token indices from 1, which the models
  take as they are (`model(inputs::NamedTuple)`: the language models, Qwen3.5 and SAM 3).
  They match
  transformers bit for bit on 8-bit images.
- An extension for ImageCore: processors take matrices of colorants, such as the images that
  `FileIO.load` returns (#348), and `Metalhead.image_matrix` turns arrays of their layout,
  such as the images that a pipeline generates and the masks of SAM 3, back into matrices of
  colorants.
- Mixed precision under `eltype = Float16` or `Flux.BFloat16` (or `f16`/`bf16`, #332): the
  RMSNorms of the language models and timm's `LayerNorm2d` keep their parameters in
  `Float32`, as Flux's `BatchNorm`, `GroupNorm` and `LayerNorm` do, and normalise in
  `Float32`, while the activations stay in half precision. ViTs resample their position
  embeddings in `Float32`, as timm does.
- Validation against PyTorch: float64 parity fixtures for every code path, run in CI without
  Python or network access, and float32 references for 71 pretrained checkpoints. Weights load
  strictly by name.
- `porting/` tools and an agent skill (`skills/port-hf-model`) for porting new
  models (#305, #307).
- Documentation: getting started, a model list grouped by family, validation, porting and a
  [migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/) from 0.9 (#307).
- A tutorial on vision models covering image preprocessing, classification, transfer learning,
  multi-scale features and vision transformer tokens.
- Tutorials on vision-language models and segmentation, a gallery and figures of the outputs
  of the models in the documentation, made by `docs/figures/make_figures.jl`, and a shorter
  README.

### Changed

- New dependencies: HuggingFaceApi, SafeTensors, JSON and ConcreteStructs, and the weak
  dependencies HuggingFaceTokenizers and ImageCore. Flux 0.16.14, NNlib 0.9.54 and MLUtils
  0.4.9 are now the minimum versions (#335, #337, #341).
- Downloads share the cache of Python's `huggingface_hub` (`~/.cache/huggingface/hub`), and
  its `HF_TOKEN`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` environment variables.

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

[0.10.0]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.7...HEAD
[0.9.7]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/FluxML/Metalhead.jl/compare/v0.9.5...v0.9.6
