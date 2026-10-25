# [Migrating from Metalhead 0.9](@id migration)

Metalhead 0.10 replaces the hand-written models with ports of timm models, built with
[`create_model`](@ref). The old API still works in 0.10 and will be removed in 0.11. The
deprecated API covers:

- the model constructors (`ResNet(50)`, `ViT(:base)`, ...) and their lowercase builders
  (`Metalhead.resnet`, ...);
- `backbone`, `classifier` and `Metalhead.loadpretrain!`;
- the `Metalhead.Layers` module;
- `pretrain = true`, and the weights it loads from the FluxML organisation on the Hub.

The model constructors, `backbone`, `classifier` and `loadpretrain!` warn once per session,
naming their replacement. The lowercase builders and `Metalhead.Layers`, which the
constructors use, don't warn.

## What changes

| 0.9 | 0.10 |
|:--|:--|
| `ResNet(50)` | `create_model("resnet50")` |
| `ResNet(50; pretrain = true)` | `create_model("timm/resnet50.tv2_in1k"; pretrained = true)` |
| `nclasses = 10` | `num_classes = 10` |
| `inchannels = 1` | `in_chans = 1` |
| `imsize = (256, 256)` | `img_size = (256, 256)` (for ViT and the MLP mixers) |
| `backbone(model)(x)` | `forward_features(model, x)` |
| `classifier(model)(features)` | `forward_head(model, features)` |
| layers from `Metalhead.Layers` | Flux layers |

The input layout (width × height × channels × batch) is unchanged. Models are now returned
in test mode.

The keyword arguments are timm's, so they differ per family. The docstrings of the old
constructors don't apply to the new models: look up the timm model instead. The models also
have a different structure (the PyTorch one), so code that indexes into a model, like
`model.layers[1][2]`, needs updating.

## Pretrained weights

The pretrained weights of 0.9 came from torchvision. timm hosts the same weights, as
`<model>.tv_in1k` (torchvision's `IMAGENET1K_V1`) and `<model>.tv2_in1k` (`IMAGENET1K_V2`).
The table gives the repository with the version that `pretrain = true` loaded. For ResNet-50,
the outputs of the two models are bitwise identical.

| 0.9 | 0.10 |
|:--|:--|
| `ResNet(18; pretrain = true)`, `ResNet(34; ...)` | `create_model("timm/resnet18.tv_in1k"; pretrained = true)`, `"timm/resnet34.tv_in1k"` |
| `ResNet(50; pretrain = true)`, `101`, `152` | `"timm/resnet50.tv2_in1k"`, `"timm/resnet101.tv2_in1k"`, `"timm/resnet152.tv2_in1k"` |
| `WideResNet(50; pretrain = true)`, `101` | `"timm/wide_resnet50_2.tv2_in1k"`, `"timm/wide_resnet101_2.tv2_in1k"` |
| `ResNeXt(50; pretrain = true)` (32×4d) | `"timm/resnext50_32x4d.tv2_in1k"` |
| `ResNeXt(101; cardinality = 32, base_width = 8, pretrain = true)` | `"timm/resnext101_32x8d.tv2_in1k"` |
| `ResNeXt(101; cardinality = 64, base_width = 4, pretrain = true)` | `"timm/resnext101_64x4d.tv_in1k"` |
| `VGG(11; pretrain = true)`, `13`, `16`, `19` | `"timm/vgg11.tv_in1k"`, `"timm/vgg13.tv_in1k"`, `"timm/vgg16.tv_in1k"`, `"timm/vgg19.tv_in1k"` |
| `ViT(...; pretrain = true)` | no torchvision weights on the Hub; use timm's, e.g. `"timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"` |
| `SqueezeNet(; pretrain = true)` | no replacement |

timm also has newer checkpoints of most of these architectures, usually more accurate. See
the [model list](@ref models).

## Architectures

| 0.9 | 0.10 |
|:--|:--|
| `ResNet(depth)` | `create_model("resnet18")`, `"resnet34"`, `"resnet50"`, `"resnet101"`, `"resnet152"` |
| `WideResNet(depth)` | `create_model("wide_resnet50_2")`, `"wide_resnet101_2"` |
| `ResNeXt(depth; cardinality, base_width)` | `create_model("resnext50_32x4d")`, `"resnext101_32x8d"`, `"resnext101_64x4d"`, ... |
| `SEResNet(depth)` | `create_model("seresnet50")`, ... |
| `SEResNeXt(depth; cardinality, base_width)` | `create_model("seresnext50_32x4d")`, `"seresnext101_32x4d"`, ... |
| `Res2Net(depth; base_width, scale)` | `create_model("res2net50_26w_4s")`, `"res2net101_26w_4s"`, ... |
| `Res2NeXt(depth; ...)` | `create_model("res2next50")` |
| `DenseNet(config)` | `create_model("densenet121")`, `"densenet161"`, `"densenet169"`, `"densenet201"` |
| `VGG(depth; batchnorm)` | `create_model("vgg16")`, or `"vgg16_bn"` with batch normalisation |
| `MobileNetv1(width_mult)` | `create_model("mobilenetv1_100")`, `"mobilenetv1_125"` |
| `MobileNetv2(width_mult)` | `create_model("mobilenetv2_100")`, `"mobilenetv2_050"`, `"mobilenetv2_140"`, ... |
| `MobileNetv3(config; width_mult)` | `create_model("mobilenetv3_small_100")`, `"mobilenetv3_large_100"`, ... |
| `MNASNet(:B1)`, `MNASNet(:A1)` | `create_model("mnasnet_100")`, `create_model("semnasnet_100")` |
| `EfficientNet(:b0)` ... `(:b8)` | `create_model("efficientnet_b0")`, or `"tf_efficientnet_b0"` with TensorFlow's padding |
| `EfficientNetv2(:small)` ... `(:xlarge)` | `create_model("tf_efficientnetv2_s")`, `"tf_efficientnetv2_m"`, `"tf_efficientnetv2_l"`, `"tf_efficientnetv2_xl"` |
| `Inceptionv3()` | `create_model("inception_v3")` |
| `Inceptionv4()` | `create_model("inception_v4")` |
| `InceptionResNetv2()` | `create_model("inception_resnet_v2")` |
| `Xception()` | no exact equivalent with Hub weights; the closest is `create_model("xception41")` |
| `ConvNeXt(:tiny)` ... `(:large)` | `create_model("convnext_tiny")`, `"convnext_small"`, `"convnext_base"`, `"convnext_large"` |
| `ConvMixer(:small)`, `ConvMixer(:base)` | `create_model("convmixer_768_32")`, `create_model("convmixer_1536_20")` |
| `ConvMixer(:large)` | no exact equivalent; the closest is `create_model("convmixer_1024_20_ks9_p14")` |
| `MLPMixer(config; patch_size)` | `create_model("mixer_s16_224")`, `"mixer_b16_224"`, `"mixer_l16_224"`, `"mixer_b32_224"`, ... |
| `ResMLP(config)` | different sizes in timm: `create_model("resmlp_12_224")`, `"resmlp_24_224"`, `"resmlp_36_224"`, `"resmlp_big_24_224"` |
| `gMLP(config)` | different sizes in timm: `create_model("gmlp_ti16_224")`, `"gmlp_s16_224"`, `"gmlp_b16_224"` |
| `ViT(:tiny)` ... `(:large)` | `create_model("vit_tiny_patch16_224")`, `"vit_small_patch16_224"`, `"vit_base_patch32_224"`, ... |
| `AlexNet`, `GoogLeNet`, `SqueezeNet`, `UNet` | no replacement: timm has no weights for them |

These architectures are not ported yet: `ConvNeXt(:xlarge)`, `ViT(:huge)`, `ViT(:giant)`
and `ViT(:gigantic)`.
