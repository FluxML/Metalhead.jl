```@meta
CollapsedDocStrings = true
```

# [Internals](@id internals)

```@index
Pages = ["internals.md"]
```

Nothing on this page is public API: none of it is exported or declared `public`, and it may
change without notice, in any release, as timm, transformers and diffusers do. Using the
models doesn't need it. The CLIP, BERT and sentence-transformers pages have their own
internals, and the [Porting](@ref porting_api) page documents how a library registers its
architectures.

## Builders

`create_model` calls these with the architecture's default arguments, merged with the
checkpoint's configuration and your keyword arguments. The timm builders take timm's keyword
arguments, the transformers builders the fields of the model's `config.json`, and the
diffusers builders the arguments of the model's constructor.

```@autodocs
Modules = [Metalhead.Timm, Metalhead.Transformers]
Order = [:function]
Public = false
Filter = f -> !(f in (Metalhead.Timm.pil_resize, Metalhead.Timm.create_conv2d,
    Metalhead.Timm.decode_arch_def, Metalhead.Transformers.chunk_gated_delta_rule,
    Metalhead.Transformers.moe_experts, Metalhead.Transformers.get_vision_features,
    Metalhead.Transformers.qwen3_5_decode, generate, Metalhead.Transformers.clip_model,
    Metalhead.Transformers.clip_text_model, Metalhead.Transformers.clip_vision_model,
    Metalhead.Transformers.clip_text_model_with_projection,
    Metalhead.Transformers.clip_vision_model_with_projection,
    Metalhead.Transformers.bert_model,
    Metalhead.Transformers.bert_for_sequence_classification,
    Metalhead.Transformers.xlm_roberta_model,
    Metalhead.Transformers.xlm_roberta_for_sequence_classification,
    Metalhead.Transformers.modernbert_model, Metalhead.Transformers.modernbert_for_masked_lm,
    Metalhead.Transformers.modernbert_for_sequence_classification))
```

```@docs
Metalhead.Diffusers.autoencoder_kl_flux2
Metalhead.Diffusers.flux2_transformer_2d_model
Metalhead.Diffusers.autoencoder_kl
Metalhead.Diffusers.z_image_transformer_2d_model
```

## Diffusers components

The transformers, the autoencoders' latent distribution and the scheduler of the pipelines:

```@docs
Metalhead.Diffusers.Flux2Transformer2DModel
Metalhead.Diffusers.ZImageTransformer2DModel
Metalhead.Diffusers.encode
Metalhead.Diffusers.decode
Metalhead.Diffusers.DiagonalGaussianDistribution
Metalhead.Diffusers.mode
Metalhead.Diffusers.sample
Metalhead.Diffusers.FlowMatchEulerDiscreteScheduler
Metalhead.Diffusers.set_timesteps
Metalhead.Diffusers.step
Metalhead.Diffusers.compute_empirical_mu
```

## Helpers of the ported models

```@docs
Metalhead.Transformers.get_vision_features
Metalhead.Transformers.qwen3_5_decode
Metalhead.Timm.EfficientNetBuilder
Metalhead.Timm.MixedConv2d
Metalhead.Timm.CondConv2d
Metalhead.Transformers.MultiheadAttention
Metalhead.Diffusers.timestep_embedding
Metalhead.Diffusers.AdaLayerNormContinuous
Metalhead.Timm.create_conv2d
Metalhead.Timm.decode_arch_def
Metalhead.Transformers.chunk_gated_delta_rule
Metalhead.Transformers.moe_experts
Metalhead.Timm.pil_resize
```

## Processors and loading

```@docs
Metalhead.checkpoint_prefix
Metalhead.checkpoint_tensor
Metalhead.load_checkpoint!
Metalhead.AbstractProcessor
Metalhead.AbstractImageProcessor
Metalhead.@processor
Metalhead.register_processor!
Metalhead.ignored_keys
Metalhead.legacy_keys
Metalhead.standardize_settings
Metalhead.instantiate_processor
Metalhead.check_inputs
Metalhead.torch_params
Metalhead.TorchParam
```
