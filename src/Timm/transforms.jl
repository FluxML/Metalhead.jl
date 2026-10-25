# timm's evaluation transform (timm 1.0.30: `resolve_data_config` in timm/data/config.py
# and `transforms_imagenet_eval` in timm/data/transforms_factory.py), and the torchvision
# 0.29 transforms it composes: `Resize`, timm's `ResizeKeepRatio`, `CenterCrop`, `ToTensor`
# and `Normalize`. timm transforms PIL images, so the port of Pillow (pil.jl) resizes them.
# `EvalTransform` is the processor of a timm checkpoint, from the `pretrained_cfg` of its
# config.json (timm/models/_hub.py and the `PretrainedCfg` of timm/models/_pretrained.py).

const IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
const IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
const DEFAULT_CROP_PCT = 0.875
const TIMM_INTERPOLATIONS = ("nearest", "bilinear", "bicubic", "box", "hamming", "lanczos")
const TIMM_CROP_MODES = ("center", "squash")

cfg_value(cfg::NamedTuple, key) = get(cfg, key, nothing)
cfg_value(cfg::AbstractDict, key) = get(cfg, String(key), nothing)

# Python's truth value, which `resolve_data_config` tests most settings by
truthy(x) = !(isnothing(x) || (x isa Number ? iszero(x) : isempty(x)))

"""
    resolve_data_config(pretrained_cfg; use_test_size = false) -> NamedTuple

The settings of timm's eval transform for the `pretrained_cfg` of a checkpoint (a `Dict`
from its `config.json`, or the `NamedTuple` of `Timm.pretrained_cfg`), resolved as timm's
`resolve_data_config({}, pretrained_cfg = cfg)` resolves them: `input_size`,
`interpolation`, `mean`, `std`, `crop_pct` and `crop_mode`, each from `pretrained_cfg` or
else timm's default, `(3, 224, 224)`, `"bicubic"`, ImageNet's statistics, `0.875` and
`"center"`. `test_input_size` and `test_crop_pct` replace `input_size` and `crop_pct` only
with `use_test_size = true`, as in timm.
"""
function resolve_data_config(cfg; use_test_size::Bool = false)
    value(key) = cfg_value(cfg, key)
    input_size = (3, 224, 224)
    if use_test_size && !isnothing(value(:test_input_size))
        input_size = Tuple(value(:test_input_size))
    elseif !isnothing(value(:input_size))
        input_size = Tuple(value(:input_size))
    end
    setting(key, default, convert) = truthy(value(key)) ? convert(value(key)) : default
    floats(v) = Tuple(Float64.(v))
    interpolation = setting(:interpolation, "bicubic", String)
    mean = setting(:mean, IMAGENET_DEFAULT_MEAN, floats)
    std = setting(:std, IMAGENET_DEFAULT_STD, floats)
    crop_pct = setting(:crop_pct, DEFAULT_CROP_PCT, Float64)
    if use_test_size && truthy(value(:test_crop_pct))
        crop_pct = Float64(value(:test_crop_pct))
    end
    crop_mode = setting(:crop_mode, "center", String)
    return (; input_size, interpolation, mean, std, crop_pct, crop_mode)
end

# An image as the `(width, height, 3)` `UInt8` array of the "RGB" PIL image that timm's
# transform takes. Integers are 8-bit values. Reals from 0 to 1 become 8-bit as
# torchvision's `ToPILImage` converts them, times 255 rounded down (which keeps `k / 255`
# at `k`). A grayscale image (a matrix, or 1 channel), gray with alpha (2) or RGBA (4) is
# converted as Pillow's `convert("RGB")` does, repeating the gray and dropping the alpha.
pil_rgb(img) = convert_to_rgb(to_uint8(image_array(img)))

to_uint8(img::AbstractArray{UInt8}) = Array(img)
function to_uint8(img::AbstractArray{<:Integer})
    all(v -> 0 <= v <= 255, img) ||
        throw(ArgumentError("an image of integers has values from 0 to 255"))
    return UInt8.(img)
end
function to_uint8(img::AbstractArray{<:Real})
    all(v -> 0 <= v <= 1, img) ||
        throw(ArgumentError("an image of real numbers has values from 0 to 1"))
    return map(v -> unsafe_trunc(UInt8, float(v) * 255), img)
end

# torchvision's `resize` of a PIL image, to `(height, width)` or its short side to
# `short_side`, the long side rounded down (`_compute_resized_output_size`).
function torchvision_resize(img, short_side::Integer, filter)
    w, h = size(img)
    short, long = minmax(w, h)
    new_long = trunc(Int, short_side * long / short)
    size_hw = w <= h ? (new_long, short_side) : (short_side, new_long)
    return torchvision_resize(img, size_hw, filter)
end
function torchvision_resize(img, (h, w)::NTuple{2, Integer}, filter)
    return (w, h) == size(img)[1:2] ? img : pil_resize(img, (w, h), filter)
end

# timm's ResizeKeepRatio in the eval transform: the short side, relative to `(h, w)`, to its
# size, the other rounded half to even. timm passes it no interpolation, so it is bilinear.
function resize_keep_ratio(img, (h, w))
    ratio = min(size(img, 2) / h, size(img, 1) / w)
    size_hw = round.(Int, (size(img, 2), size(img, 1)) ./ ratio)
    return torchvision_resize(img, size_hw, :bilinear)
end

# torchvision's `center_crop`: zero padding where the image is smaller than the crop, and
# the offsets rounded half to even.
function center_crop(img, (h, w))
    img_w, img_h, c = size(img)
    if w > img_w || h > img_h
        left, top = max(w - img_w, 0) ÷ 2, max(h - img_h, 0) ÷ 2
        padded = zeros(UInt8, max(w, img_w), max(h, img_h), c)
        padded[left .+ (1:img_w), top .+ (1:img_h), :] = img
        img = padded
        img_w, img_h = size(img)
    end
    left, top = round(Int, (img_w - w) / 2), round(Int, (img_h - h) / 2)
    return img[left .+ (1:w), top .+ (1:h), :]
end

function check_transform(interpolation, crop_mode, mean, std)
    interpolation in TIMM_INTERPOLATIONS ||
        throw(ArgumentError("interpolation = $(repr(interpolation)) is not supported \
                             (supported: $(join(repr.(TIMM_INTERPOLATIONS), ", ")))"))
    crop_mode in TIMM_CROP_MODES ||
        throw(ArgumentError("crop_mode = $(repr(crop_mode)) is not supported (supported: \
                             $(join(repr.(TIMM_CROP_MODES), ", ")))"))
    length(mean) == length(std) == 3 ||
        throw(ArgumentError("`mean` and `std` have one value per RGB channel"))
    return nothing
end

"""
    eval_transform(images; input_size = (3, 224, 224), interpolation = "bicubic",
                   crop_pct = 0.875, crop_mode = "center", mean = IMAGENET_DEFAULT_MEAN,
                   std = IMAGENET_DEFAULT_STD) -> Array{Float32, 4}

timm's eval transform, `create_transform(; settings...)`, of one image or a vector of
images, as a `(width, height, channels, batch)` array, bit for bit. The defaults are those
of `resolve_data_config`, not `create_transform`'s own (`"bilinear"`).

An image is a `(width, height, channels)` array (torch's `(channels, height, width)`
reversed), as the PIL image that timm transforms: `UInt8`, or other integers, from 0 to
255; or reals from 0 to 1, converted to 8 bits as torchvision's `ToPILImage` converts them
(times 255, rounded down). It has 3 channels (RGB), or 1 (or is a matrix), 2 or 4,
converted as Pillow's `convert("RGB")` converts L, LA and RGBA images.

The short side of each image is resized to `floor(input_size / crop_pct)` with Pillow's
`interpolation` filter (`"nearest"`, `"bilinear"`, `"bicubic"`, `"box"`, `"hamming"` or
`"lanczos"`), then center-cropped to `input_size` (padded with zeros where it is smaller),
scaled to [0, 1] and normalized with `mean` and `std`. `crop_mode = "squash"` resizes both
sides to `floor(input_size / crop_pct)` instead, ignoring the aspect ratio. With a
non-square `input_size` and `crop_mode = "center"`, timm resizes with `"bilinear"`,
whatever `interpolation` says. timm's third crop mode, `"border"`, is not supported.
"""
function eval_transform(images; input_size = (3, 224, 224), interpolation = "bicubic",
        crop_pct = DEFAULT_CROP_PCT, crop_mode = "center", mean = IMAGENET_DEFAULT_MEAN,
        std = IMAGENET_DEFAULT_STD)
    interpolation = string(interpolation)
    check_transform(interpolation, crop_mode, mean, std)
    h, w = input_size isa Integer ? (input_size, input_size) :
           Tuple(input_size)[(end - 1):end]
    filter = Symbol(interpolation)
    crop_pct = truthy(crop_pct) ? crop_pct : DEFAULT_CROP_PCT
    scale_h, scale_w = floor(Int, h / crop_pct), floor(Int, w / crop_pct)
    images = image_batch(images)
    out = Array{Float32}(undef, w, h, 3, length(images))
    for (i, image) in enumerate(images)
        x = pil_rgb(image)
        x = if crop_mode == "squash"
            torchvision_resize(x, (scale_h, scale_w), filter)
        elseif scale_h == scale_w
            torchvision_resize(x, scale_h, filter)
        else
            resize_keep_ratio(x, (scale_h, scale_w))
        end
        x = center_crop(x, (h, w))
        for c in 1:3
            m, s = Float32(mean[c]), Float32(std[c])
            out[:, :, c, i] .= (Float32.(view(x, :, :, c)) ./ 255.0f0 .- m) ./ s
        end
    end
    return out
end

"""
    Timm.EvalTransform(; input_size = (3, 224, 224), interpolation = "bicubic",
                       mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225),
                       crop_pct = 0.875, crop_mode = "center")

timm's evaluation transform, `create_transform(**resolve_data_config(...))`: the image
processor of every timm checkpoint, which [`create_processor`](@ref
Metalhead.create_processor) builds from the `pretrained_cfg` of its config.json. Its
settings are those that timm's `resolve_data_config` resolves, with the same defaults, which
apply where the configuration doesn't set them.

Called on one image or a vector of images, it returns the input of the model, not a
NamedTuple: a `Float32` array of size `(W, H, 3, batch)`, where `(3, H, W)` is
`input_size`, of the images resized, cropped and normalized, bit for bit as timm transforms
them as PIL images. One image gives a batch of one.

```julia
model = create_model("timm/convnext_tiny.fb_in1k"; pretrained = true)
transform = create_processor("timm/convnext_tiny.fb_in1k")
logits = model(transform(image))    # transform(image): 224 × 224 × 3 × 1
```

An image is an array of size `(width, height, channels)` (torch's
`(channels, height, width)` reversed) of integers from 0 to 255 (an 8-bit image, as `UInt8`)
or of reals from 0 to 1, which are converted to 8 bits as torchvision's `ToPILImage`
converts them, or, with ImageCore loaded, a matrix of colorants (see
[`image_array`](@ref Metalhead.image_array)). Grey images, and the alpha of RGBA, are
converted to RGB as Pillow's `convert("RGB")` does. Each image is

 1. resized with Pillow's filter `interpolation` (`"nearest"`, `"bilinear"`, `"bicubic"`,
    `"box"`, `"hamming"` or `"lanczos"`): its short side to `floor(input_size / crop_pct)`
    with `crop_mode = "center"`, both sides with `crop_mode = "squash"`. (A non-square
    `input_size` with `"center"` resizes with `"bilinear"`, as in timm.) timm's third crop
    mode, `"border"`, is not supported;
 2. center-cropped to the height and width of `input_size` `(channels, height, width)`,
    padded with zeros where it is smaller;
 3. scaled to [0, 1] and normalized with `mean` and `std`.

`create_processor(repo; use_test_size = true)` resolves `test_input_size` and
`test_crop_pct` in place of `input_size` and `crop_pct`, where the configuration has them,
as `resolve_data_config(use_test_size = True)` does: many checkpoints are evaluated at a
larger size than the one they were trained at. The other keywords of `create_processor`
override the settings, e.g. `crop_pct = 1.0`. [`Timm.pretrained_cfg`](@ref
Metalhead.Timm.pretrained_cfg) returns the `pretrained_cfg` as it is.
"""
Base.@kwdef struct EvalTransform <: ImageProcessor
    input_size::NTuple{3, Int} = (3, 224, 224)
    interpolation::String = "bicubic"
    mean::Tuple{Vararg{Float64}} = IMAGENET_DEFAULT_MEAN
    std::Tuple{Vararg{Float64}} = IMAGENET_DEFAULT_STD
    crop_pct::Float64 = DEFAULT_CROP_PCT
    crop_mode::String = "center"
    provenance::NamedTuple = (;)
end

function (t::EvalTransform)(images)
    return eval_transform(images; t.input_size, t.interpolation, t.mean, t.std, t.crop_pct,
        t.crop_mode)
end

function check_processor(t::EvalTransform)
    return check_transform(t.interpolation, t.crop_mode, t.mean, t.std)
end

# The keys of timm's `PretrainedCfg` that the eval transform doesn't read.
function ignored_keys(::Type{EvalTransform})
    weights = "says where timm finds the weights"
    classifier = "describes the classifier"
    model = "describes the model"
    metadata = "metadata"
    return (; url = weights, file = weights, state_dict = weights, hf_hub_id = weights,
        hf_hub_filename = weights, source = weights,
        custom_load = "says how timm loads the weights",
        architecture = "names the architecture", tag = "names the pretrained weights",
        test_input_size = "replaces `input_size` with `use_test_size = true`",
        test_crop_pct = "replaces `crop_pct` with `use_test_size = true`",
        min_input_size = "the smallest input size of the model, not one of the transform",
        fixed_input_size = "a setting of the model, which `create_model` reads",
        num_classes = classifier, label_offset = classifier, label_names = classifier,
        label_descriptions = classifier, pool_size = model, test_pool_size = model,
        first_conv = model, classifier = model, license = metadata, description = metadata,
        origin_url = metadata, paper_name = metadata, paper_ids = metadata,
        notes = metadata)
end

# The settings that `resolve_data_config` takes from the `pretrained_cfg` only where they are
# true in Python, so that an empty string or list, or zero, is the default.
const TRUTHY_SETTINGS = ("interpolation", "mean", "std", "crop_pct", "crop_mode",
    "test_crop_pct")

# The `pretrained_cfg` of a config.json; a former config.json is the `pretrained_cfg` with the
# architecture (timm's `_parse_model_cfg`).
function timm_pretrained_cfg(config)
    haskey(config, "pretrained_cfg") && return config["pretrained_cfg"]
    outside = ("architecture", "num_features", "labels")
    return Dict(k => v for (k, v) in config if !(k in outside))
end

function instantiate_processor(::TimmFormat, files, config; use_test_size::Bool = false,
        kwargs...)
    entries, test = Tuple{String, Any, String}[], Tuple{String, Any, String}[]
    for (k, v) in timm_pretrained_cfg(config)
        k in TRUTHY_SETTINGS && !truthy(v) && continue
        if use_test_size && k in ("test_input_size", "test_crop_pct")
            push!(test, (replace(k, "test_" => ""), jsonvalue(v), "config.json (`$k`)"))
        else
            push!(entries, (k, jsonvalue(v), "config.json"))
        end
    end
    return processor_from_entries(EvalTransform, [entries; test]; kwargs...)
end
