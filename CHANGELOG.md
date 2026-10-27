# Changelog

All notable changes to Metalhead.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases before 0.9.6 are described
in the [GitHub releases](https://github.com/FluxML/Metalhead.jl/releases).

## [Unreleased]

### Added

- The text encoders BERT, XLM-RoBERTa and ModernBERT (`BertModel`, `XLMRobertaModel` and
  `ModernBertModel`, with `BertForSequenceClassification`,
  `XLMRobertaForSequenceClassification`, `ModernBertForSequenceClassification` and
  `ModernBertForMaskedLM`), the backbones of embedding models, rerankers and masked language
  models, validated against `sentence-transformers/all-MiniLM-L6-v2`,
  `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2`,
  `intfloat/multilingual-e5-base`, `BAAI/bge-reranker-base`, `BAAI/bge-reranker-v2-m3`,
  `Alibaba-NLP/gte-modernbert-base`, `Alibaba-NLP/gte-reranker-modernbert-base` and
  `answerdotai/ModernBERT-base`. A model maps token indices (and an `attention_mask` for a
  batch padded on the right, and BERT's `token_type_ids`) to the hidden states of the tokens;
  `forward_head(model, hidden)` is the `pooler_output` of BERT and XLM-RoBERTa, and
  `forward_intermediates` returns the output of each layer. XLM-RoBERTa numbers its
  positions after the padding, ModernBERT alternates local attention in a window of 128
  tokens with global attention, with a rotary base for each. Pooling the hidden states into
  one embedding per text is not part of them. `BAAI/bge-m3` has no safetensors weights and
  doesn't load.
- `TextProcessor` returns the `token_type_ids` of the tokenizers that transformers gives
  them (BERT's), as its new setting `model_input_names` says, and encodes pairs of texts
  with `text_pair`, joined by the template of the tokenizer: the input of a reranker. Its
  tokenizer no longer truncates the texts when its tokenizer.json sets a truncation
  (`sentence-transformers/all-MiniLM-L6-v2`), as transformers' tokenizers don't by default.
- SigLIP and SigLIP 2, Google's image-text dual encoders (`SiglipModel` and `Siglip2Model`),
  for zero-shot classification and image-text retrieval, validated against
  `google/siglip-base-patch16-224`, `google/siglip2-base-patch16-224` and
  `google/siglip2-base-patch16-naflex`: their text and vision models, with the
  attention-pooling head, and the normalized embeddings and the logits of the sigmoid
  similarities (`logits_per_image`, `logits_per_text`) that they return.
  `Metalhead.Transformers.get_text_features` and `get_image_features` encode texts or images
  alone. The NaFlex checkpoints of SigLIP 2 (`Siglip2Model`) take images of any aspect ratio
  as patches, and resize their position embeddings to the grid of each image; the others
  are `SiglipModel`s. Their processors, `SiglipProcessor` and `Siglip2Processor`, with
  `SiglipImageProcessor` and `Siglip2ImageProcessor`, pad the texts to 64 tokens, as the
  models were trained, and the NaFlex processor lowercases them. A transformers
  `config.json` without `architectures`, as those of the SigLIP 2 checkpoints of fixed
  resolution, names its model by its `model_type`.
- A tutorial on zero-shot classification and image-text retrieval with SigLIP and SigLIP 2.
- CLIP, the dual encoder of images and texts (`CLIPModel`), for zero-shot classification and
  image-text retrieval, validated against `openai/clip-vit-base-patch32`,
  `openai/clip-vit-large-patch14` and `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (QuickGELU and
  GELU), with its processor, `Metalhead.Transformers.CLIPProcessor`, whose
  `CLIPImageProcessor` resizes the shorter side of an image and crops its center. The model
  returns transformers' `logits_per_image`, `logits_per_text` and normalized embeddings, and
  `get_text_features` and `get_image_features` the outputs of its encoders. The encoders load
  alone too (`CLIPTextModel`, validated against the text encoder of Stable Diffusion 1.5,
  `CLIPVisionModel`, and both `...WithProjection`), from checkpoints saved by transformers 4
  or 5, and the vision encoder takes images of other sizes with
  `interpolate_pos_encoding = true`.
- Qwen3 MoE (`Qwen3MoeForCausalLM`, Qwen3-30B-A3B) and Qwen3.5 MoE
  (`Qwen3_5MoeForConditionalGeneration`, Qwen3.5-35B-A3B and Qwen3.6-35B-A3B, and the text
  model `Qwen3_5MoeForCausalLM`): Qwen3 and Qwen3.5 whose MLPs are sparse mixtures of experts,
  with a shared expert in Qwen3.5 MoE, validated against the first 4 decoder layers of
  `Qwen/Qwen3-30B-A3B` and `Qwen/Qwen3.6-35B-A3B` (with its vision encoder, on text and on an
  image), whose whole checkpoints, of 61 and 72 GB, are too large for the machine of the
  checks. `porting/dump_reference.py --layers 4 --selective-shards` writes such references,
  which download only the shards of the layers kept. The Qwen3.5 MoE checkpoints that transformers 5
  saves, with one MLP per expert rather than Qwen's stacked experts, don't load.
- Text-to-image generation with Z-Image (`ZImagePipeline`, with `ZImageTransformer2DModel`
  and `AutoencoderKL`, diffusers' autoencoder of Stable Diffusion and FLUX.1), validated
  against `Tongyi-MAI/Z-Image-Turbo`. Its sampling loop has diffusers' classifier-free
  guidance with a negative prompt, `cfg_normalization` and `cfg_truncation`; Z-Image-Turbo
  samples in 9 steps with `guidance_scale = 0`. Its text encoder is the decoder of the
  `Qwen3ForCausalLM` checkpoint, as diffusers loads a `Qwen3Model`, whose prompt embeddings
  are a matrix per prompt, without the padding.

### Fixed

- Pretrained models allocate their weights directly in the requested `eltype`, without
  random initialization or a temporary Float32 model. Mixed precision normalization and
  random initialization of replacement classifiers are preserved (#366).

## [0.10.0]

Breaking: Flux 0.16.14, NNlib 0.9.55 and MLUtils 0.4.9 are now the minimum versions, and the
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
- Qwen3.8 (`Qwen/Qwen3.8-27B`), a `Qwen3_5ForConditionalGeneration` checkpoint, validated
  on text and on an image in float32 with the weights of its embeddings, vision encoder,
  head and first 8 of its 64 decoder layers. Its `output_gate_type` (`"swish"`), which
  transformers ignores, gating the linear attention with SiLU, must be `"silu"` or `"swish"`.
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
- Image editing with FLUX.2 [klein]: `Flux2KleinPipeline` takes reference images, `image`
  (one image or a vector of them, as processors take images), and generates the image that
  the prompt asks of them. As diffusers does, it prepares each (an image over 1024 × 1024
  pixels resized with Pillow's Lanczos filter, the sides cropped to multiples of 16), encodes
  it with the autoencoder and lets the transformer attend to its tokens; the image takes the
  size of the first reference by default. Validated against diffusers with tiny pipelines and
  `black-forest-labs/FLUX.2-klein-4B` (#365).
- Text embeddings of [sentence-transformers](https://github.com/huggingface/sentence-transformers)
  models, in `Metalhead.SentenceTransformers`, validated against `Qwen/Qwen3-Embedding-0.6B`.
  `create_model` builds a `SentenceTransformer` from a repository with a `modules.json`,
  whatever its `config.json` names: the modules of its pipeline (`Transformer`, which holds
  the base model of a ported transformers architecture and its tokenizer; `Pooling`, with
  every pooling mode, the left padding and `include_prompt`; `Dense`; `Normalize`) and the
  prompts, similarity function and `truncate_dim` of its
  `config_sentence_transformers.json`, read strictly, as are the `max_seq_length` and
  `do_lower_case` of `sentence_bert_config.json`, in the formats of sentence-transformers
  before and after 5. `encode(model, texts)` (a method of HuggingFaceTokenizers' `encode`),
  `encode_query`, `encode_document` and `similarity`, exported, compute and compare
  embeddings as sentence-transformers does, and `model(inputs)` takes the inputs of
  `Metalhead.SentenceTransformers.preprocess`. `create_processor` of such a repository is an
  error, as the model prepares its inputs (#374).
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
  dependencies HuggingFaceTokenizers and ImageCore. Flux 0.16.14, NNlib 0.9.55 and MLUtils
  0.4.9 are now the minimum versions (#335, #337, #341).
- Downloads share the cache of Python's `huggingface_hub` (`~/.cache/huggingface/hub`), and
  its `HF_TOKEN`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` environment variables.
- `create_model` and `create_processor` look for a `modules.json` before the `config.json`
  of a repository, which costs a request to the Hub for a repository without one.

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
