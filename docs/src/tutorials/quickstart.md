# [Getting started](@id quickstart)

For a worked example with photographs, a custom image dataset and transfer learning, see
[Working with vision models](@ref vision_models).

## Creating a model

[`create_model`](@ref) builds a ported model. Its first argument names the model in one of
three ways:

| Argument | Example | Meaning |
|:--|:--|:--|
| Hub repository | `"timm/resnet50.a1_in1k"` | the architecture and weights of that repository |
| architecture | `"resnet50"` | the bare architecture; with `pretrained = true`, its default checkpoint |
| local directory | `"path/to/dir"` | a `config.json` and `model.safetensors` laid out like a Hub repository |

```julia
using Metalhead

model = create_model("resnet50")                                 # random weights, no download
model = create_model("timm/resnet50.a1_in1k")                   # that repository's architecture, random weights
model = create_model("timm/resnet50.a1_in1k"; pretrained = true) # weights from the Hub
```

Weights are random unless `pretrained = true`. Then Metalhead downloads and caches the
repository's `config.json` and safetensors weights (sharded checkpoints too). The cache is shared with Python's
`huggingface_hub` (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE`), and so are `HF_TOKEN` for
private or gated repositories and `HF_HUB_OFFLINE` to use only cached files.
The [model list](@ref models) shows the checkpoints each architecture was validated against.
Bare architecture names use the first of them and load it at the commit it was validated at.
Other repositories load from their `main` branch, or from `revision`:

```julia
model = create_model("timm/resnet50.a1_in1k"; pretrained = true, revision = "main")
```

The other keyword arguments are the original library's model arguments, such as timm's
`num_classes`, `drop_rate` or `drop_path_rate`. They override the repository's configuration.
Use `eltype` to build the model in another precision (`Float16`, `Flux.BFloat16`, `Float32` or
`Float64`).

[`list_models`](@ref) lists the ported architectures, or with `pretrained = true` the
validated checkpoints:

```julia
list_models()
list_models(; pretrained = true)
```

Use `byfamily = true` to return a dictionary keyed by family identifiers, such as `"resnet"`
and `"xception_aligned"`:

```julia
families = list_models(; byfamily = true)
keys(families)
families["resnet"]
list_models(; byfamily = true, pretrained = true)["resnet"]
```

Browse the [model list](@ref models) for architectures grouped by family and links to their
validated checkpoints.

## Inputs

Image models take Flux's layout: width × height × channels × batch, in `Float32` unless you chose
another `eltype`. PyTorch's `(N, C, H, W)` is the same memory layout with the dimensions
reversed. Sizes in timm configurations, such as `input_size = (3, 224, 224)`, are in PyTorch
order. The processor of a checkpoint makes these inputs from images (see below).

The model is returned in test mode: dropout is off and batch normalisation uses its running
statistics. Call `Flux.trainmode!(model)` before training.

## Classifying an image

A checkpoint expects its images prepared as in its evaluation: resized, cropped and
normalised with the settings of its configuration. [`create_processor`](@ref) reads them
from the same repository and revision as the weights, and returns the processor of the
checkpoint, which applies them. For a timm checkpoint, it is timm's eval transform, a
[`Timm.EvalTransform`](@ref Metalhead.Timm.EvalTransform):

```julia
using Metalhead, Flux, FileIO, ImageCore

repo = "timm/convnext_tiny.fb_in1k"
model = create_model(repo; pretrained = true)
transform = create_processor(repo)

img = load(download("https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"))
x = transform(img)                # 224 × 224 × 3 × 1
probs = softmax(model(x))
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))
labels[sortperm(vec(probs); rev = true)[1:5]]
# "Labrador retriever", "kelpie", "dingo", "golden retriever", "Chesapeake Bay retriever"
```

FileIO reads the photo with ImageIO, which must be installed, as a height × width matrix of
colours. With ImageCore loaded (Images.jl loads it too), processors take such matrices, and
turn them into the width × height × channels arrays of Flux. Without it, pass an array of
that layout, of `UInt8` from 0 to 255 or of reals from 0 to 1. A vector of images makes a
batch.

The transform resizes the shorter side of the image to `floor(224 / 0.875) = 256` pixels
with Pillow's bicubic filter, crops the centre 224 × 224 pixels, and normalises each channel
with `mean` and `std`. These are the values that timm computes from the same pixels as a PIL
image, bit for bit (the pixels of a JPEG decoded by ImageIO can differ slightly from
Pillow's). `show` prints each setting and where it comes from:

```julia
transform
# EvalTransform
#   input_size = (3, 224, 224)    config.json
#   interpolation = "bicubic"     config.json
#   mean = (0.485, 0.456, 0.406)  config.json
#   std = (0.229, 0.224, 0.225)   config.json
#   crop_pct = 0.875              config.json
#   crop_mode = "center"          config.json
```

Many timm checkpoints are evaluated at a larger size than they were trained at, given by
`test_input_size` and `test_crop_pct` in their configuration. `use_test_size = true` takes
these, as timm's `resolve_data_config(use_test_size = True)` does. Other keyword arguments
override a setting by name, also in natural forms (`input_size = 256` for
`(3, 256, 256)`, `mean = 0.5` for every channel, `interpolation = :bilinear`):

```julia
transform = create_processor(repo; use_test_size = true)
size(transform(img))              # (288, 288, 3, 1)
transform = create_processor(repo; crop_pct = 1.0)
transform.crop_pct                # 1.0, shown as "override (config.json: 0.875)"
```

A setting that the port doesn't implement, such as timm's `crop_mode = "border"`, is an
error, and so is a keyword that isn't a setting. [`Metalhead.Timm.pretrained_cfg`](@ref)
returns the settings as the configuration stores them.

## Features and the classifier

Image models follow timm's split into a feature extractor and a head:

```julia
x = rand(Float32, 224, 224, 3, 2)
features = forward_features(model, x)                       # e.g. 7 × 7 × 768 × 2 for ConvNeXt-T
pooled = forward_head(model, features; pre_logits = true)   # 768 × 2
logits = forward_head(model, features)                      # 1000 × 2
```

## Multi-scale features

Segmentation and depth heads take the feature maps of several stages. `forward_intermediates`
returns them, like timm's, and `feature_info` gives their channels and strides:

```julia
resnet = create_model("resnet50")
feature_info(resnet)              # (num_chs = 64, reduction = 2, module_name = "act1"), ...
maps = forward_intermediates(resnet, x; indices = 2:5, intermediates_only = true)
size.(maps)                       # 56 × 56 × 256 × 2, 28 × 28 × 512 × 2, ...
```

`indices` are 1-based positions in `feature_info`, or an integer `k` for the last `k`.
`stop_early = true` stops at the last selected tap; `norm = true` applies a final norm
where the architecture supports it. Convolutional models accept `output_fmt = "NCHW"`
(WHCN arrays in Julia).

Feature extraction covers ResNet and its variants (including Res2Net), ConvNeXt, ViT,
the MLP mixers, DenseNet, VGG, Inception v3/v4, Inception-ResNet-v2, aligned Xception,
and all ported EfficientNet and MobileNet families (including MixNet, MNASNet, FBNet
and LCNet). ConvMixer has no upstream feature tap table and is unsupported. The
`create_model(...; features_only = true, out_indices = ...)` convenience interface is
not implemented; use `forward_intermediates` on the existing model.

Taps follow timm's definitions: VGG includes a reduction-1 tap, ConvNeXt starts at
reduction 4, and Xception taps inside its strided blocks before downsampling. Dilated
EfficientNet and Xception models can have several taps with the same reduction.

Vision transformers return the tokens of their blocks as feature maps of the patch grid
(`output_fmt = "NLC"` keeps them as tokens), and with `dynamic_img_size = true` accept any
input size that is a multiple of the patch size:

```julia
dinov2 = create_model("vit_small_patch14_dinov2"; pretrained = true, dynamic_img_size = true)
maps = forward_intermediates(dinov2, rand(Float32, 224, 308, 3, 1); indices = 4,
    intermediates_only = true)    # 16 × 22 × 384 × 1 each
```

## Fine-tuning

Pass a new `num_classes` to load the pretrained weights with a freshly initialised classifier:

```julia
using Flux

model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true, num_classes = 10)
Flux.trainmode!(model)

loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
data = [(rand(Float32, 224, 224, 3, 4), rand(1:10, 4))]  # labels in 1:num_classes
opt_state = Flux.setup(AdamW(1e-4), model)
Flux.train!(loss, model, data, opt_state)
```

With `num_classes = 0` the model has no classifier and returns the pooled features.

## Model structure

A ported model mirrors the PyTorch module tree: its fields have the names of the PyTorch
submodules, and `nn.Sequential`s are `Chain`s with Julia index `i` for PyTorch index `i - 1`.
PyTorch's `model.layer1[0].conv1` is `model.layer1[1].conv1`:

```julia
model = create_model("resnet50")
model.layer1[1].conv1     # Conv((1, 1), 64 => 64, bias=false)
model.fc                  # Dense(2048 => 1000)
```

The layers are ordinary Flux layers, so `Flux.freeze!`, `Flux.state` and `Flux.loadmodel!` work
as usual.

## GPU

Load [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and cuDNN, or another GPU backend, and move
the model and its inputs to the device. See the
[Flux GPU guide](https://fluxml.ai/Flux.jl/stable/guide/gpu/).

```julia
using Flux, Metalhead, CUDA, cuDNN

device = gpu_device()
model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true) |> device
logits = model(device(rand(Float32, 224, 224, 3, 8)))
```
