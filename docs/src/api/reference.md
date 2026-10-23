# API reference

## Models

`create_model` and `list_models` apply to every ported model. The other functions are the
interface of image backbones and classifiers; language models have `forward_features`,
`forward_head` and `forward_intermediates` too.

```@docs
create_model
list_models
forward_features
forward_head
forward_intermediates
feature_info
```

## timm

```@docs
Metalhead.Timm.pretrained_cfg
```

## diffusers

```@docs
Metalhead.Diffusers.Flux2KleinPipeline
Metalhead.Diffusers.tokenize
Metalhead.Diffusers.encode_prompt
Metalhead.Diffusers.Flux2Transformer2DModel
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

## Builders

`create_model` calls these with the architecture's default arguments, merged with the
checkpoint's configuration and your keyword arguments. The timm builders take timm's keyword
arguments, the transformers builders the fields of the model's `config.json`, and the
diffusers builders the arguments of the model's constructor.

```@autodocs
Modules = [Metalhead.Timm, Metalhead.Transformers]
Order = [:function]
Filter = f -> f !== Metalhead.Timm.pretrained_cfg
```

```@docs
Metalhead.Diffusers.autoencoder_kl_flux2
Metalhead.Diffusers.flux2_transformer_2d_model
```

## Porting internals

These are used by the ported models and the porting tools. Using the models doesn't need them.

```@docs
Metalhead.Timm
Metalhead.Timm.EfficientNetBuilder
Metalhead.Timm.MixedConv2d
Metalhead.Timm.CondConv2d
Metalhead.Transformers
Metalhead.Diffusers
Metalhead.Diffusers.timestep_embedding
Metalhead.Diffusers.AdaLayerNormContinuous
Metalhead.ArchSpec
Metalhead.ConfigFormat
Metalhead.architecture_of
Metalhead.model_args
Metalhead.skip_prefix
Metalhead.weights_name
Metalhead.instantiate
Metalhead.Checkpoint
Metalhead.load_state!
Metalhead.torch_params
Metalhead.TorchParam
```
