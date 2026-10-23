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
```

### Creating a model

`create_model` takes the name of a model in one of three forms:

| Argument | Example | Meaning |
|:--|:--|:--|
| Hub repository | `"timm/resnet50.a1_in1k"` | the architecture and weights of that repository |
| architecture | `"resnet50"` | the bare architecture; with `pretrained = true`, its default checkpoint |
| local directory | `"path/to/dir"` | a `config.json` and `model.safetensors` laid out like a Hub repository |

```julia
model = create_model("resnet50")                                  # random weights, no download
model = create_model("timm/resnet50.a1_in1k")                     # that repository's architecture, random weights
model = create_model("timm/resnet50.a1_in1k"; pretrained = true)  # weights from the Hub
model = create_model("resnet50"; drop_path_rate = 0.1)            # timm's model arguments

list_models()                     # the ported architectures
list_models(; pretrained = true)  # the checkpoints validated against PyTorch
list_models(; byfamily = true)    # a dictionary of architectures by family
list_models(; byfamily = true)["resnet"]
```

Any Hub repository whose architecture has been ported can be loaded, not only the validated
checkpoints. With `pretrained = true`, the repository's `config.json` and safetensors weights
are downloaded once and cached in the cache of Python's `huggingface_hub`
(`~/.cache/huggingface/hub`, or `HF_HUB_CACHE`). `HF_TOKEN` gives access to private or gated
repositories, and `HF_HUB_OFFLINE=1` uses only cached files. The keyword `revision` picks a
branch, tag or commit, and `eltype` builds the model in `Float16`, `Flux.BFloat16`, `Float32`
(the default) or `Float64`. Other keywords are the model arguments of the original library
(timm's `num_classes`, `drop_rate`, ...), and override the repository's configuration.

### Inputs and outputs

Image models, which are all the ported models so far, take Flux's layout, width × height ×
channels × batch. PyTorch's `(N, C, H, W)` is the same memory layout with the dimensions
reversed, and sizes in timm configurations, such as `input_size = (3, 224, 224)`, are in
PyTorch order. Classifiers return logits, of size classes × batch.

Models are returned in test mode: dropout is off and batch normalisation uses its running
statistics. Call `Flux.trainmode!(model)` before training.

### Preprocessing

The preprocessing a checkpoint was trained with is in its configuration:

```julia
cfg = Metalhead.Timm.pretrained_cfg("timm/convnext_tiny.fb_in1k")
cfg.input_size     # (3, 224, 224)
cfg.crop_pct       # 0.875
cfg.mean, cfg.std  # ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
```

Resize the shorter side of the image to `input_size / crop_pct`, crop the centre, and normalise
each channel with `mean` and `std`. The
[getting started guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/) does this
with Images.jl to classify a photo.

### Features and the classifier

Image models follow timm's split into a feature extractor and a head:

```julia
model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true)
x = rand(Float32, 224, 224, 3, 2)

features = forward_features(model, x)                      # last feature map, 7 × 7 × 768 × 2
pooled = forward_head(model, features; pre_logits = true)  # pooled features, 768 × 2
logits = forward_head(model, features)                     # same as model(x), 1000 × 2
```

### Multi-scale features

Segmentation and depth heads take the feature maps of several stages. `forward_intermediates`
returns them, as timm's does, and `feature_info` gives their channels and strides:

```julia
maps = forward_intermediates(model, x; intermediates_only = true)  # 56 × 56 × 96 × 2, ...
feature_info(model)  # (num_chs = 96, reduction = 4, module_name = "stages.0"), ...
```

### Fine-tuning

A new `num_classes` loads the pretrained weights with a freshly initialised classifier, and
`num_classes = 0` removes the classifier, so that the model returns the pooled features:

```julia
using Flux

model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true, num_classes = 10)
Flux.trainmode!(model)

loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
data = [(rand(Float32, 224, 224, 3, 4), Flux.onehotbatch(rand(1:10, 4), 1:10))]
opt_state = Flux.setup(AdamW(1e-4), model)
Flux.train!(loss, model, data, opt_state)
```

### Model structure

A ported model mirrors the PyTorch module tree. Its fields have the names of the PyTorch
submodules, and an `nn.Sequential` is a `Chain`, with Julia index `i` for PyTorch index `i - 1`:
PyTorch's `model.layer1[0].conv1` is `model.layer1[1].conv1`.

```julia
model = create_model("resnet50")
model.layer1[1].conv1  # Conv((1, 1), 64 => 64, bias=false)
model.fc               # Dense(2048 => 1000)
```

The layers are ordinary Flux layers, so `Flux.freeze!`, `Flux.state` and `Flux.loadmodel!` work
as usual, and models move to a GPU with Flux's `gpu_device()`.

See the [getting started guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/) for
more, and the [list of models](https://fluxml.ai/Metalhead.jl/dev/models/) for the ported
architectures and their validated checkpoints.

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

## Acknowledgements

The architectures, pretrained weights and their configurations come from
[timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models), by Ross
Wightman and the Hugging Face team. Metalhead's ports translate timm's code, and its tests
compare against timm's outputs. Pretrained weights keep the license of their Hub repository.

Porting timm with AI agents, loading the Hub weights by name and checking the result against
PyTorch, follows [Luximm.jl](https://github.com/csvance/Luximm.jl) by Carroll Vance, which
does the same for [Lux.jl](https://lux.csail.mit.edu).
