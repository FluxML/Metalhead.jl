# API reference

## Models

`create_model` and `list_models` apply to every ported model. The other functions are the
interface of image backbones and classifiers, which are all the ported models so far.

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

## Builders

`create_model` calls these with the architecture's default arguments, merged with the
checkpoint's configuration and your keyword arguments. The timm builders take timm's keyword
arguments, and the transformers builders the fields of the model's `config.json`.

```@autodocs
Modules = [Metalhead.Timm, Metalhead.Transformers]
Order = [:function]
Filter = f -> f !== Metalhead.Timm.pretrained_cfg
```

## Porting internals

These are used by the ported models and the porting tools. Using the models doesn't need them.

```@docs
Metalhead.Timm
Metalhead.Timm.EfficientNetBuilder
Metalhead.Timm.MixedConv2d
Metalhead.Timm.CondConv2d
Metalhead.Transformers
Metalhead.ArchSpec
Metalhead.ConfigFormat
Metalhead.architecture_of
Metalhead.model_args
Metalhead.skip_prefix
Metalhead.Checkpoint
Metalhead.load_state!
Metalhead.torch_params
Metalhead.TorchParam
```
