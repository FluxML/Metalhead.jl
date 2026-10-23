# [Getting started](@id quickstart)

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

The other keyword arguments are the original library's model arguments (timm's, for now), such
as `num_classes`, `drop_rate` or `drop_path_rate`. They override the repository's configuration. Use `eltype` to build the
model in another precision (`Float16`, `Float32` or `Float64`).

[`list_models`](@ref) lists the ported architectures, or with `pretrained = true` the
validated checkpoints:

```julia
list_models()
list_models(; pretrained = true)
```

Browse the [model list](@ref models) for architectures grouped by family and links to their
validated checkpoints.

## Inputs

Image models take Flux's layout: width × height × channels × batch, in `Float32` unless you chose
another `eltype`. PyTorch's `(N, C, H, W)` is the same memory layout with the dimensions
reversed. Sizes in timm configurations, such as `input_size = (3, 224, 224)`, are in PyTorch
order.

The model is returned in test mode: dropout is off and batch normalisation uses its running
statistics. Call `Flux.trainmode!(model)` before training.

## Classifying an image

The preprocessing a checkpoint expects (image size, crop, normalisation) is in its
configuration; [`Metalhead.Timm.pretrained_cfg`](@ref) returns it. The steps below follow timm's
evaluation transform: resize the shorter side to `input_size / crop_pct`, crop the centre, and
normalise. They use [Images.jl](https://github.com/JuliaImages/Images.jl).

```julia
using Metalhead, Flux, Images

repo = "timm/convnext_tiny.fb_in1k"
model = create_model(repo; pretrained = true)
cfg = Metalhead.Timm.pretrained_cfg(repo)

function preprocess(img, cfg)
    _, h, w = cfg.input_size
    scale = floor(Int, h / cfg.crop_pct) / minimum(size(img))
    img = imresize(img; ratio = scale)
    top, left = (size(img) .- (h, w)) .÷ 2
    img = img[(top + 1):(top + h), (left + 1):(left + w)]
    x = permutedims(Float32.(channelview(RGB.(img))), (3, 2, 1))  # C × H × W to W × H × C
    x = (x .- reshape(Float32.(collect(cfg.mean)), 1, 1, 3)) ./
        reshape(Float32.(collect(cfg.std)), 1, 1, 3)
    return reshape(x, size(x)..., 1)
end

img = load(download("https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"))
probs = softmax(model(preprocess(img, cfg)))
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))
labels[sortperm(vec(probs); rev = true)[1:5]]
```

`channelview` gives channels × height × width, so the permutation is needed. Without it the
model silently sees a transposed image.

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
data = [(rand(Float32, 224, 224, 3, 4), Flux.onehotbatch(rand(1:10, 4), 1:10))]
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
