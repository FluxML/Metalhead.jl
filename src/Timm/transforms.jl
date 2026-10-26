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

# Python's truth value, by which timm tests most settings
truthy(x) = !(isnothing(x) || (x isa Number ? iszero(x) : isempty(x)))

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

"""
    Timm.EvalTransform(; input_size = (3, 224, 224), interpolation = "bicubic",
                       mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225),
                       crop_pct = 0.875, crop_mode = "center")

timm's evaluation transform, `create_transform(**resolve_data_config(...))`: the image
processor of every timm checkpoint, which [`create_processor`](@ref
Metalhead.create_processor) builds from the `pretrained_cfg` of its config.json, resolved as
timm's `resolve_data_config` resolves it: a setting that the configuration doesn't have, or
that is false in Python (`""`, `[]`, `0`), takes the default above.

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
converts them (see [`image_array`](@ref Metalhead.image_array)). Grey images, and the alpha
of RGBA, are converted to RGB as Pillow's `convert("RGB")` does. Each image is

 1. resized with Pillow's filter `interpolation` (`"nearest"`, `"bilinear"`, `"bicubic"`,
    `"box"`, `"hamming"` or `"lanczos"`): its short side to `floor(input_size / crop_pct)`
    with `crop_mode = "center"`, both sides with `crop_mode = "squash"`. (A non-square
    `input_size` with `"center"` resizes with `"bilinear"`, as in timm.) timm's third crop
    mode, `"border"`, is not supported;
 2. center-cropped to the height and width of `input_size` `(channels, height, width)`,
    padded with zeros where it is smaller;
 3. scaled to [0, 1] and normalized with `mean` and `std`.

The settings also take the forms that timm's `create_transform` takes: `input_size` as
`(height, width)` or an integer, the side of a square; `mean` and `std` as one value for
all channels; `interpolation` and `crop_mode` as symbols; and `crop_pct` and `crop_mode` as
`nothing` for their defaults.

`create_processor(repo; use_test_size = true)` resolves `test_input_size` and
`test_crop_pct` in place of `input_size` and `crop_pct`, where the configuration has them,
as `resolve_data_config(use_test_size = True)` does: many checkpoints are evaluated at a
larger size than the one they were trained at. The other keywords of `create_processor`
override the settings, e.g. `crop_pct = 1.0`. [`Timm.pretrained_cfg`](@ref
Metalhead.Timm.pretrained_cfg) returns the `pretrained_cfg` as it is.
"""
@processor struct EvalTransform <: AbstractImageProcessor
    input_size::NTuple{3, Int} = (3, 224, 224)
    interpolation::String = "bicubic"
    mean::NTuple{3, Float64} = IMAGENET_DEFAULT_MEAN
    std::NTuple{3, Float64} = IMAGENET_DEFAULT_STD
    crop_pct::Float64 = DEFAULT_CROP_PCT
    crop_mode::String = "center"
    provenance::NamedTuple = (;)
end

function standardize_settings(::Type{EvalTransform}, s::NamedTuple)
    interpolation = timm_name(:interpolation, s.interpolation, TIMM_INTERPOLATIONS)
    crop_mode = timm_name(:crop_mode, something(s.crop_mode, "center"), TIMM_CROP_MODES)
    return (; input_size = timm_input_size(s.input_size), interpolation,
        mean = timm_stats(:mean, s.mean), std = timm_stats(:std, s.std),
        crop_pct = timm_crop_pct(s.crop_pct), crop_mode)
end

function timm_input_size(x)
    size = x isa Integer ? (3, x, x) :
           x isa Union{Tuple, AbstractVector} && length(x) == 2 ? (3, x...) :
           x isa Union{Tuple, AbstractVector} && length(x) == 3 ? Tuple(x) : nothing
    size !== nothing && all(v -> v isa Integer && v > 0, size) && return Int.(size)
    throw(ArgumentError("`input_size = $(repr(x))`: `input_size` is `(channels, height, width)`, `(height, width)` or an integer, the side of a square"))
end

function timm_name(name, x, supported)
    x isa Union{AbstractString, Symbol} ||
        throw(ArgumentError("`$name = $(repr(x))`: `$name` is a string or a symbol"))
    String(x) in supported && return String(x)
    throw(ArgumentError("`$name = $(repr(String(x)))` is not supported (supported: \
                         $(join(repr.(supported), ", ")))"))
end

# One value is repeated, as `resolve_data_config` repeats it and `Normalize` broadcasts it
function timm_stats(name, x)
    v = x isa Real ? (x,) : x isa Union{Tuple, AbstractVector} ? Tuple(x) : nothing
    v !== nothing && length(v) in (1, 3) && all(e -> e isa Real, v) &&
        return length(v) == 1 ? ntuple(_ -> Float64(only(v)), 3) : Float64.(v)
    throw(ArgumentError("`$name = $(repr(x))`: `$name` has one value per RGB channel, or one for all"))
end

# `transforms_imagenet_eval` takes `crop_pct or DEFAULT_CROP_PCT`
function timm_crop_pct(x)
    (x === nothing || x isa Real && iszero(x)) && return DEFAULT_CROP_PCT
    x isa Real && !(x isa Bool) && x > 0 && return Float64(x)
    throw(ArgumentError("`crop_pct = $(repr(x))`: `crop_pct` is a positive number, the fraction of the resized image that the crop keeps"))
end

function (t::EvalTransform)(images)
    _, h, w = t.input_size
    filter = Symbol(t.interpolation)
    scale_h, scale_w = floor(Int, h / t.crop_pct), floor(Int, w / t.crop_pct)
    images = image_batch(images)
    out = Array{Float32}(undef, w, h, 3, length(images))
    for (i, image) in enumerate(images)
        x = pil_rgb(image)
        x = if t.crop_mode == "squash"
            torchvision_resize(x, (scale_h, scale_w), filter)
        elseif scale_h == scale_w
            torchvision_resize(x, scale_h, filter)
        else
            resize_keep_ratio(x, (scale_h, scale_w))
        end
        x = center_crop(x, (h, w))
        for c in 1:3
            m, s = Float32(t.mean[c]), Float32(t.std[c])
            out[:, :, c, i] .= (Float32.(view(x, :, :, c)) ./ 255.0f0 .- m) ./ s
        end
    end
    return out
end

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

# timm's `resolve_data_config({}, pretrained_cfg = cfg, use_test_size)`, over the defaults of
# `EvalTransform`
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
