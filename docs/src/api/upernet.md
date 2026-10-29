# UperNet

The heads of transformers' `UperNetForSemanticSegmentation`, ported with their module names,
so that the `decode_head.*` and `auxiliary_head.*` weights of its checkpoints, such as
`openmmlab/upernet-convnext-tiny`, load strictly by name. They take the feature maps of a
backbone (W, H, channels, batch), fine to coarse. The backbone and the image processor of
UperNet's checkpoints are not ported yet, nor is `create_model` support for them.

```@docs
Metalhead.Transformers.UperNetHead
Metalhead.Transformers.UperNetFCNHead
Metalhead.Transformers.UperNetConvModule
Metalhead.Transformers.UperNetPyramidPoolingBlock
```
