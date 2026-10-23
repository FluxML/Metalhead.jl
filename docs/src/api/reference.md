# API reference

## Models

```@docs
create_model
list_models
Metalhead.pretrained_cfg
forward_features
forward_head
forward_intermediates
feature_info
```

## Builders

`create_model` calls these with the architecture's default arguments, merged with the
checkpoint's configuration and your keyword arguments. They take timm's keyword arguments.

```@autodocs
Modules = [Metalhead.Timm]
Order = [:function]
```

## Porting internals

These are used by the ported models and the porting tools. Using the models doesn't need them.

```@docs
Metalhead.Timm
Metalhead.Timm.EfficientNetBuilder
Metalhead.Timm.MixedConv2d
Metalhead.Timm.CondConv2d
Metalhead.ArchSpec
Metalhead.Checkpoint
Metalhead.load_state!
Metalhead.torch_params
Metalhead.TorchParam
```
