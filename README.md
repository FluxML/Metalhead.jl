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
  Qwen, Gemma 3 and Granite, and the Qwen3.5 vision-language models, which also read images;
  and SAM 3, which segments the objects of an image that a text or box prompts describe.
- **Text-to-image pipelines** from [diffusers](https://github.com/huggingface/diffusers):
  FLUX.2 [klein].

The [list of models](https://fluxml.ai/Metalhead.jl/dev/models/) gives every ported
architecture and the Hub checkpoints validated for it.

`create_model` downloads a repository's configuration and weights and returns an ordinary
Flux model, ready for inference or fine-tuning, on the CPU or a GPU. `create_processor`
loads the processor of the same repository, which turns images, text and boxes into the
inputs of the model, and its outputs back into results. The models run in Julia, without
PyTorch. Each port mirrors the module tree of the PyTorch original, loads its safetensors
weights by name, and is
[checked numerically against PyTorch](https://fluxml.ai/Metalhead.jl/dev/validation/), and
so is each processor.

## Contents

- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
  - [Creating a model](#creating-a-model)
  - [Inputs and outputs](#inputs-and-outputs)
  - [Processors](#processors)
  - [Features and the classifier](#features-and-the-classifier)
  - [Multi-scale features](#multi-scale-features)
  - [Fine-tuning](#fine-tuning)
  - [Model structure](#model-structure)
  - [Language models](#language-models)
  - [Vision-language models](#vision-language-models)
  - [Segmenting objects](#segmenting-objects)
  - [Text-to-image generation](#text-to-image-generation)
- [Validated against PyTorch](#validated-against-pytorch)
- [Scope](#scope)
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
| [transformers](https://github.com/huggingface/transformers) | Llama, Qwen2 and Qwen3, Gemma 3, and Granite, with its mixtures of experts and the Granite 4.0 models without Mamba layers; Qwen3.5, a vision-language model; SAM 3 and SAM 3.1, which segment the objects that a prompt describes |
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
using Metalhead, FileIO, ImageCore

# Download the configuration and weights of a Hub repository
model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true)
# and the preprocessing of the checkpoint: timm's eval transform, with its settings
transform = create_processor("timm/convnext_tiny.fb_in1k")

img = load("dog.jpg")                                      # height × width matrix of colours
image = permutedims(rawview(channelview(img)), (3, 2, 1))  # width × height × channels, UInt8
x = transform(image)   # 224 × 224 × 3 × 1: width × height × channels × batch
logits = model(x)      # 1000 × 1
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

### Processors

A checkpoint expects its inputs prepared in a particular way, with settings stored in its
repository: the image size, crop and normalisation, the tokenizer. `create_processor` reads
them, from the same repository and revision as `create_model`, and returns the processor of
the checkpoint, which applies them as its Python library does. It reads the files of a
branch at one commit, that of the first file it finds, and offline or without network those
of the cache. `create_model` makes the forward pass match PyTorch; `create_processor` makes
the whole path from raw input to result match.

| Checkpoints | `create_processor` returns | Called as |
|:--|:--|:--|
| timm | `Timm.EvalTransform`, timm's eval transform | `transform(images)` |
| Qwen3.5 | `Transformers.Qwen3VLProcessor` | `proc(; images, text)` |
| SAM 3 | `Transformers.Sam3Processor` | `proc(; images, text, input_boxes)` |
| language models | the tokenizer, HuggingFaceTokenizers' `Tokenizer` | `encode(tok, text)` |

timm's transform returns an array of size width × height × channels × batch. The others
return a NamedTuple with the keys of transformers (`input_ids`, `pixel_values`, ...), in
Julia's layout, with token indices that start at 1, and SAM 3's turns the outputs of the
model back into results: `post_process_instance_segmentation(proc, outputs)` for masks. A
diffusers pipeline has no processor: it prepares its inputs itself, with its tokenizer
`pipe.tokenizer`. Images are arrays of size width × height × channels, of `UInt8` from 0 to
255 or of reals from 0 to 1. The matrix of colours `img` that `FileIO.load` returns becomes
one with ImageCore: `permutedims(rawview(channelview(img)), (3, 2, 1))`.

Metalhead ports no tokenizer. The tokenizer of a repository is a `Tokenizer` of
[HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl), which
runs Hugging Face's tokenizers library on the repository's `tokenizer.json`, with the padding
token of its `tokenizer_config.json`. `create_processor` loads it after
`using HuggingFaceTokenizers`, alone for a language model, or as the `tokenizer` of a
processor that takes text, such as those of Qwen3.5 and SAM 3.

The settings are read strictly: one that the port doesn't implement is an error, and an
unknown key in a file a warning. `show` prints each setting and where it comes from, and
keyword arguments override them, in the forms of the Python library or in natural ones
(`input_size = 256`, `mean = 0.5`, `interpolation = :bilinear`):

```julia
create_processor("timm/convnext_tiny.fb_in1k"; crop_pct = 1.0)
# EvalTransform
#   input_size = (3, 224, 224)    config.json
#   interpolation = "bicubic"     config.json
#   mean = (0.485, 0.456, 0.406)  config.json
#   std = (0.229, 0.224, 0.225)   config.json
#   crop_pct = 1.0                override (config.json: 0.875)
#   crop_mode = "center"          config.json
```

For timm checkpoints, `use_test_size = true` resizes to the larger `test_input_size` at
which many are evaluated, and `Metalhead.Timm.pretrained_cfg` returns the settings as they
are stored. The
[getting started guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/quickstart/) classifies
a photo.

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
vocabulary × sequence length (× batch). Their processor is the tokenizer of the repository,
HuggingFaceTokenizers.jl's, whose token ids start at 0:

```julia
using HuggingFaceTokenizers

model = create_model("HuggingFaceTB/SmolLM2-135M"; pretrained = true)
tok = create_processor("HuggingFaceTB/SmolLM2-135M")

ids = encode(tok, "The capital of France is the city of").ids .+ 1   # ids start at 0
logits = model(ids)                                      # vocabulary × sequence length
decode(tok, [argmax(logits[:, end]) - 1])                # " Paris"
```

Texts of different lengths make a batch padded on the right, with an `attention_mask` that
is 0 for the padding: `model(ids; attention_mask)` (see the
[language models guide](https://fluxml.ai/Metalhead.jl/dev/tutorials/language_models/)).

### Vision-language models

Qwen3.5 answers prompts about images. Its processor resizes each image to sides that are
multiples of 32 pixels, cuts it into the patches of 16 × 16 pixels that the model takes, and
replaces the image placeholder `<|image_pad|>` of the prompt with one image token for each
2 × 2 patches. It doesn't apply the chat template: the prompt is written as the template
writes it.

```julia
using FileIO, ImageCore, HuggingFaceTokenizers

model = create_model("Qwen/Qwen3.5-0.8B"; pretrained = true)
proc = create_processor("Qwen/Qwen3.5-0.8B")

prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" *
         "Describe this image.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
image = permutedims(rawview(channelview(load("dog.jpg"))), (3, 2, 1))
inputs = proc(; images = image, text = prompt)           # 396 tokens for 36 × 44 patches
logits = model(inputs)                                   # vocabulary × sequence length
decode(proc.tokenizer, [argmax(logits[:, end]) - 1])    # "This"

(; input_ids, pixel_values, image_grid_thw) = inputs
for _ in 1:20                                            # greedy decoding
    push!(input_ids, argmax(model(input_ids; pixel_values, image_grid_thw)[:, end]))
end
decode(proc.tokenizer, input_ids[(end - 19):end] .- 1)
# "This image features a medium-to-large dog standing in profile on a grassy field. The
#  dog has"
```

A vector of prompts makes a batch, padded on the right with an `attention_mask`, which the
model takes as it is.

### Segmenting objects

SAM 3 finds and segments the objects of an image that a text prompt names, or that box
prompts show. Its processor resizes the images to the model's 1008 × 1008 pixels and pads
the prompts to 32 tokens, and its post-processing keeps the objects found, with their boxes
and masks at the size of the image:

```julia
using FileIO, ImageCore, HuggingFaceTokenizers
using Metalhead.Transformers: post_process_instance_segmentation

model = create_model("unsloth/sam3.1"; pretrained = true)   # SAM 3.1; SAM 3: "jetjodh/sam3"
proc = create_processor("unsloth/sam3.1")

image = permutedims(rawview(channelview(load("dog.jpg"))), (3, 2, 1))
inputs = proc(; images = image, text = "a dog")
objects = only(post_process_instance_segmentation(proc, model(inputs); threshold = 0.5,
    target_sizes = inputs.original_sizes))
objects.scores, objects.boxes, objects.masks
# [0.94], 4 × 1 boxes (x1, y1, x2, y2) and 700 × 577 × 1 masks, for a photo of 700 × 577
```

A box prompt is `(x1, y1, x2, y2)` in the pixels of its image, a vector of them per image:
`proc(; images, input_boxes = [[(56, 38, 647, 550)]])`. The prompts need the tokenizer of the
processor, which `create_processor` loads after `using HuggingFaceTokenizers`; without it, the
processor prepares the images only.

Both repositories are ungated; `jetjodh/sam3` holds the same weights as Meta's gated
`facebook/sam3`. Only the detector of SAM 3's video model is ported, not the tracker of
objects across frames.

### Text-to-image generation

A diffusers repository holds a pipeline, whose components `create_model` builds from the
subfolders of the repository, its tokenizer too. With HuggingFaceTokenizers.jl loaded
first, it takes prompts:

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

## Scope

Metalhead ports what a checkpoint determines, and checks it against PyTorch. It doesn't
reimplement the general-purpose machinery of timm, transformers and diffusers. A feature
belongs in Metalhead when both of these hold:

1. The checkpoint determines it: through its configuration, its weights, or upstream code
   specific to its architecture.
2. Getting it wrong would silently change the results compared with the Python library, so
   it has to be validated against them.

In scope:

- Architectures, their configurations and weights, and the feature interface of image
  models (`forward_features`, `forward_intermediates`, ...).
- The inference preprocessing that a checkpoint's configuration sets: resizing, cropping,
  normalising, and the packing specific to a model, such as image patches and image
  placeholder tokens.
- The post-processing specific to a model, such as SAM 3's masks and boxes.
- The schedulers and sampling loops of the ported diffusion pipelines. The sampler, not the
  denoiser alone, defines the image. A scheduler is ported along with a pipeline that uses
  it, not on its own.
- The key-value cache of language models (not implemented yet).

Out of scope, for other packages:

- Tokenization algorithms. HuggingFaceTokenizers.jl tokenizes, and Metalhead uses it through
  an extension.
- Chat templates, and decoding strategies such as beam search, logits processors and
  streaming.
- Training: optimizers, learning-rate schedules, losses, weight averaging and training loops,
  which Flux, Optimisers.jl and ParameterSchedulers.jl provide. The models are ordinary
  differentiable Flux models.
- Data augmentation, data loading and image I/O, which DataAugmentation.jl, MLUtils.jl and
  FileIO.jl provide.
- Task APIs such as transformers' `pipeline`, quantization, LoRA and distributed training.

General-purpose code that matching PyTorch requires, such as PIL's resampling filters, stays
internal rather than public. Architectures are added once a checkpoint validates them; the
goal is not to cover each library's whole model collection.

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
