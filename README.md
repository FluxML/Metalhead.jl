# Metalhead

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.github.io/Metalhead.jl/dev)
[![CI](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/FluxML/Metalhead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Metalhead.jl)

Metalhead.jl provides [Flux.jl](https://fluxml.ai) ports of models from the
[Hugging Face Hub](https://huggingface.co/models), starting with the vision models of
[timm](https://github.com/huggingface/pytorch-image-models). Each port mirrors the module
structure of the PyTorch original, loads the original weights directly from the Hub, and is
checked numerically against PyTorch.

## Installation

Metalhead.jl requires Julia 1.10 or later. Install it from the Julia package manager:

```julia
julia> ]add Metalhead
```

## Usage

```julia
using Metalhead

# Download the configuration and weights of a Hub repository
model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true)

x = rand(Float32, 224, 224, 3, 1)   # width × height × channels × batch
logits = model(x)                   # 1000 × 1

features = forward_features(model, x)                              # last feature map
pooled = forward_head(model, features; pre_logits = true)          # pooled, before the classifier

list_models()                                                      # the ported architectures
create_model("resnet50")                                           # random weights, no download
create_model("resnet50"; pretrained = true, num_classes = 10)      # new classifier, to fine-tune
```

Any Hub repository whose architecture has been ported can be loaded, not only the checkpoints
listed in the documentation. See the
[getting started guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/) for
preprocessing images and fine-tuning, and the
[list of models](https://fluxml.ai/Metalhead.jl/dev/models/) for the ported architectures.

## Validated against PyTorch

A model is added to Metalhead only after it [reproduces PyTorch](https://fluxml.ai/Metalhead.jl/dev/validation/):

- A small, randomly initialised version of each architecture matches PyTorch's float64 outputs
  to a relative error of 1e-8. These checks run in CI without Python or network access.
- With real pretrained weights, outputs match PyTorch to 1e-4 in float32.
- Weights load by name, strictly: every tensor in the checkpoint must be used, and every
  parameter of the model must be found, with the right shape.

## Contributing

New models are ported with the tools in
[`porting/`](https://github.com/FluxML/Metalhead.jl/tree/master/porting) and a
[Claude Code](https://claude.com/claude-code) skill that takes an agent through a port and its
validation. See [Porting a model](https://fluxml.ai/Metalhead.jl/dev/contributing/).

## Upgrading from Metalhead 0.9

Metalhead 0.10 deprecates the hand-written models (`ResNet(50)`, `ViT(:base)`, ...), the
`Metalhead.Layers` module and `pretrain = true`. They will be removed in 0.11. The
[migration guide](https://fluxml.ai/Metalhead.jl/dev/migration/) gives the `create_model`
equivalent of each model.
