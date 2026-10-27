# CLIP

See the [zero-shot classification tutorial](@ref zero_shot) for using the model and its
processor together.

## Models

```@docs
Metalhead.Transformers.CLIPModel
Metalhead.Transformers.CLIPTextModel
Metalhead.Transformers.CLIPVisionModel
Metalhead.Transformers.CLIPTextModelWithProjection
Metalhead.Transformers.CLIPVisionModelWithProjection
```

## Builders

`create_model` calls these with the fields of the `config.json` of a checkpoint.

```@docs
Metalhead.Transformers.clip_model
Metalhead.Transformers.clip_text_model
Metalhead.Transformers.clip_vision_model
Metalhead.Transformers.clip_text_model_with_projection
Metalhead.Transformers.clip_vision_model_with_projection
```

## Processors

```@docs
Metalhead.Transformers.CLIPProcessor
Metalhead.Transformers.CLIPImageProcessor
```
