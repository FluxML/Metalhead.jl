# Metalhead

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.github.io/Metalhead.jl/dev)
[![CI](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/Metalhead.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/FluxML/Metalhead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Metalhead.jl)

Metalhead.jl brings pretrained models from the [Hugging Face Hub](https://huggingface.co/models)
to [Flux.jl](https://fluxml.ai). It ports models from three Hugging Face libraries:

- **Vision models** from [timm](https://github.com/huggingface/pytorch-image-models): about
  300 architectures, including ResNet, ConvNeXt, EfficientNet, MobileNet and ViT, for
  classification or as backbones.
- **Language models** from [transformers](https://github.com/huggingface/transformers): Llama,
  Qwen, Gemma 3 and Granite, and the Qwen3.5 vision-language models, which also read images.
- **Text-to-image pipelines** from [diffusers](https://github.com/huggingface/diffusers):
  FLUX.2 [klein].

The [list of models](https://fluxml.ai/Metalhead.jl/dev/models/) gives every ported
architecture and the Hub checkpoints validated for it.

`create_model` downloads a repository's configuration and weights and returns an ordinary Flux
model, ready for inference or fine-tuning, on the CPU or a GPU. The models run in Julia,
without PyTorch. Each port mirrors the module tree of the PyTorch original, loads its
safetensors weights by name, and is
[checked numerically against PyTorch](https://fluxml.ai/Metalhead.jl/dev/validation/).

## Contents

- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
  - [Creating a model](#creating-a-model)
  - [Inputs and outputs](#inputs-and-outputs)
  - [Preprocessing](#preprocessing)
  - [Features and the classifier](#features-and-the-classifier)
  - [Multi-scale features](#multi-scale-features)
  - [Fine-tuning](#fine-tuning)
  - [Model structure](#model-structure)
  - [Language models](#language-models)
  - [Vision-language models](#vision-language-models)
  - [Text-to-image generation](#text-to-image-generation)
- [Validated against PyTorch](#validated-against-pytorch)
- [Contributing](#contributing)
- [Upgrading from Metalhead 0.9](#upgrading-from-metalhead-09)
- [Acknowledgements](#acknowledgements)

## Installation

Metalhead.jl requires Julia 1.10 or later. Install it from the Julia package manager:

```julia
julia> ]add Metalhead
```

## Models

| Library | Ported families |
|:--|:--|
| [timm](https://github.com/huggingface/pytorch-image-models) | ResNet, ResNeXt, Wide ResNet, SE- and ECA-ResNet, Res2Net, ConvNeXt, DenseNet, VGG, Inception, Inception-ResNet, Xception, EfficientNet and EfficientNetV2, MobileNetV1 to V4, MNASNet, MixNet, FBNet, LCNet, ConvMixer, MLP-Mixer, ResMLP, gMLP, and ViT with the image encoders of DINOv2, CLIP and SigLIP |
| [transformers](https://github.com/huggingface/transformers) | Llama, Qwen2 and Qwen3, Gemma 3, and Granite, with its mixtures of experts and the Granite 4.0 models without Mamba layers; Qwen3.5, a vision-language model |
| [diffusers](https://github.com/huggingface/diffusers) | FLUX.2 [klein]: the text-to-image pipeline, its transformer and its autoencoder |

The [list of models](https://fluxml.ai/Metalhead.jl/dev/models/) gives each architecture of
these families and the Hub checkpoints whose outputs were compared with PyTorch's. Other Hub
repositories of the same architectures load too. From Julia:

```julia
list_models()                             # the ported architectures
list_models(; pretrained = true)          # the validated checkpoints
list_models(; byfamily = true)["resnet"]  # the architectures of a family
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
| component of a pipeline | `"black-forest-labs/FLUX.2-klein-4B/vae"` | the model in that subfolder of a diffusers repository |

```julia
model = create_model("resnet50")                                  # random weights, no download
model = create_model("timm/resnet50.a1_in1k")                     # that repository's architecture, random weights
model = create_model("timm/resnet50.a1_in1k"; pretrained = true)  # weights from the Hub
model = create_model("resnet50"; drop_path_rate = 0.1)            # timm's model arguments
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

Image models take Flux's layout, width × height × channels × batch. PyTorch's `(N, C, H, W)` is the same memory layout with the dimensions
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
data = [(rand(Float32, 224, 224, 3, 4), rand(1:10, 4))]  # labels in 1:num_classes
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

### Language models

Language models take token indices, which start at 1, and return logits, of size
vocabulary × sequence length (× batch). Metalhead has no tokenizers:
[HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl) loads
them from the Hub.

```julia
using HuggingFaceTokenizers

model = create_model("HuggingFaceTB/SmolLM2-135M"; pretrained = true)
tokenizer = from_pretrained(Tokenizer, "HuggingFaceTB/SmolLM2-135M")

ids = encode(tokenizer, "The capital of France is the city of").ids .+ 1   # ids start at 0
logits = model(ids)                                      # vocabulary × sequence length
decode(tokenizer, [argmax(logits[:, end]) - 1])          # " Paris"
```

### Vision-language models

Qwen3.5 answers prompts about images. `qwen3_5_image_inputs` resizes and cuts an image into
the patches the model takes, and the prompt holds one image token for each 2 × 2 patches:

```julia
using HuggingFaceTokenizers
using Metalhead.Transformers: qwen3_5_image_inputs

model = create_model("Qwen/Qwen3.5-0.8B"; pretrained = true)
tokenizer = from_pretrained(Tokenizer, "Qwen/Qwen3.5-0.8B")

image = rand(Float32, 640, 480, 3)    # width × height × channels, in [0, 1]
pixel_values, image_grid_thw = qwen3_5_image_inputs(image)
n = prod(image_grid_thw) ÷ 4                              # the number of image tokens
prompt = "<|im_start|>user\n<|vision_start|>" * "<|image_pad|>"^n *
         "<|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n" *
         "<think>\n\n</think>\n\n"
ids = encode(tokenizer, prompt).ids .+ 1
for _ in 1:20                                             # greedy decoding
    push!(ids, argmax(model(ids; pixel_values, image_grid_thw)[:, end]))
end
decode(tokenizer, ids[(end - 19):end] .- 1)
```

### Text-to-image generation

A diffusers repository holds a pipeline, whose components `create_model` builds from the
subfolders of the repository. With HuggingFaceTokenizers.jl loaded, it takes prompts:

```julia
using Flux, HuggingFaceTokenizers

pipe = create_model("black-forest-labs/FLUX.2-klein-4B"; pretrained = true, eltype = BFloat16)
image = gpu(pipe)("A photo of a red fox sitting in fresh snow"; num_inference_steps = 4)
size(image)                                              # 1024 × 1024 × 3 × 1, values in [0, 1]
```

See the [getting started guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/) for
more, the [language models guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/language_models/)
for generating text, the
[text-to-image guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/text_to_image/) for
generating images, and the [list of models](https://fluxml.ai/Metalhead.jl/dev/models/) for
the ported architectures and their validated checkpoints.

The [vision models tutorial](https://fluxml.ai/Metalhead.jl/dev/tutorials/vision/) walks through
classifying a photograph, training a classifier on your images and fine-tuning a backbone.

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

Metalhead's ports translate the code of three Hugging Face libraries, and its tests compare
against their outputs: [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch
Image Models), by Ross Wightman and the Hugging Face team, and
[transformers](https://github.com/huggingface/transformers) and
[diffusers](https://github.com/huggingface/diffusers), by the Hugging Face team and their
contributors. The pretrained weights and their configurations are those of the Hub
repositories, published by timm or by each model's authors, and keep the license of their
repository.

Porting models with AI agents, loading the Hub weights by name and checking the result
against PyTorch, follows [Luximm.jl](https://github.com/csvance/Luximm.jl) by Carroll Vance,
which does the same for timm in [Lux.jl](https://lux.csail.mit.edu).

Before the ports, Metalhead was a collection of hand-written Flux models.
[Mike Innes](https://github.com/MikeInnes) started it in 2018, with
[Avik Pal](https://github.com/avik-pal), and [Dhairya Gandhi](https://github.com/DhairyaLGandhi)
and [Kyle Daruwalla](https://github.com/darsnack) maintained it through the following years.
[Abhirath Anand](https://github.com/theabhirath) wrote most of the
models and layers of Metalhead 0.9, and [Brian Chen](https://github.com/ToucheSir) kept its
tests running on CPUs and GPUs. [Carlo Lucibello](https://github.com/CarloLucibello) ported
the pretrained weights of torchvision to the Hugging Face Hub in 2023, has maintained
Metalhead since, and rebuilt it around ports of Hub models in Metalhead 0.10. Many
[other contributors](https://github.com/FluxML/Metalhead.jl/graphs/contributors) added models,
fixes and documentation.
