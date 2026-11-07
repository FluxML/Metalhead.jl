# API reference

```@docs
Metalhead
```

Everything on the pages of this reference is public, exported or declared with `public` on
Julia 1.11 and later, except the [Internals](@ref internals) and [Porting](@ref porting_api)
pages and the sections called Internals. The fields of model structs mirror PyTorch's and are
not public, but for the documented `generation_config` of the language models.

## Models

`create_model` and `list_models` apply to every ported model. The other functions are the
interface of image backbones and classifiers, and of language models, which `generate` text.

```@docs
create_model
list_models
forward_features
forward_head
forward_intermediates
feature_info
generate
Metalhead.Transformers.GenerationConfig
```

## Processors

[`create_processor`](@ref) loads the processor of a checkpoint, which turns raw inputs into
the inputs of its model and its outputs back into results. Each processor type ports a class
of the Python library of its checkpoints, with its settings as fields.

```@docs
create_processor
Metalhead.TextProcessor
Metalhead.Timm.EvalTransform
Metalhead.Transformers.Qwen3VLProcessor
Metalhead.Transformers.Qwen2VLImageProcessor
```

Post-processing takes the processor and the outputs of the model:

```@docs
post_process_object_detection
post_process_instance_segmentation
post_process_semantic_segmentation
```

Image processors take images in one layout (and so do pipelines, their reference images), to
which an extension for another type of image adds a method, and `image_matrix` turns arrays
of that layout, such as generated images, back into images of colorants:

```@docs
Metalhead.image_array
Metalhead.image_matrix
```

## timm

```@docs
Metalhead.Timm
Metalhead.Timm.pretrained_cfg
```

## transformers

```@docs
Metalhead.Transformers
Metalhead.Transformers.KVCache
Metalhead.Transformers.last_tokens
Metalhead.Transformers.get_text_features
Metalhead.Transformers.get_image_features
Metalhead.Transformers.Qwen3_5ForConditionalGeneration
Metalhead.Transformers.Qwen3_5VisionModel
```

The models and processors of CLIP, of the text encoders BERT, XLM-RoBERTa and ModernBERT, of
SAM 3, and of SigLIP and SigLIP 2, have their own API references:
[CLIP](clip.md), [BERT, XLM-RoBERTa and ModernBERT](bert.md), [SAM 3](sam3.md), and
[SigLIP and SigLIP 2](siglip.md).

## diffusers

```@docs
Metalhead.Diffusers
Metalhead.Diffusers.Flux2KleinPipeline
Metalhead.Diffusers.encode_prompt
Metalhead.Diffusers.ZImagePipeline
```

## sentence-transformers

`create_model` builds a `SentenceTransformer` from a repository with a `modules.json`. Its
classes and functions have their own [API reference](sentence_transformers.md).

```@docs
Metalhead.SentenceTransformers
```

## Internals

The [Internals](@ref internals) page documents what is not public: the builders, the
components of the pipelines and the helpers of the ported models, which may change without
notice. The [Porting](@ref porting_api) page documents how a library registers its
architectures, also internal.
