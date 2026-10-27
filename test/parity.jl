# Numerical parity of Metalhead ports against PyTorch, on the fixtures written by porting/dump.py
# and porting/dump_reference.py. Used by the test suite and by porting/check.jl.
module Parity

using Metalhead, Flux, Functors, LinearAlgebra, SafeTensors
using Metalhead: Checkpoint, join_key, layer_params

export Tap, tapify, check_fixture, fixture_model, fixture_inputs, shrink_channels,
       check_lm_reference, check_image_reference, check_sam3_reference, check_pipeline_fixture,
       reference_model, check_siglip_reference, check_diffusers_reference,
       reference_input, relerr

relerr(a, b) = norm(a .- b) / max(norm(b), floatmin(Float64))

const TAP_RTOL = Dict(Float16 => 2e-3, Float32 => 1e-5, Float64 => 1e-8)

"""
    reference_input(torch_shape)

Deterministic input matching `reference_input` in porting/common.py, in Julia layout.
"""
function reference_input(torch_shape)
    x = Array{Float32}(undef, reverse(torch_shape)...)
    for I in CartesianIndices(x)
        w, h, c, b = Tuple(I) .- 1
        x[I] = Float32(mod(b * 131 + c * 71 + h * 37 + w * 17, 97) / 24 - 2)
    end
    return x
end

"""
    Tap(module, name, sink)

Wraps a module of a ported model, recording each output in `sink[name]`.
"""
struct Tap{M}
    _m::M
    _name::String
    _sink::Dict{String, Vector{Any}}
end
function (t::Tap)(args...; kws...)
    y = getfield(t, :_m)(args...; kws...)
    push!(get!(getfield(t, :_sink), getfield(t, :_name), Any[]), y)
    return y
end
Base.getproperty(t::Tap, s::Symbol) = getproperty(getfield(t, :_m), s)
Flux.@layer Tap

"""
    tapify(model, names, sink) -> model

Rebuild `model` with a [`Tap`](@ref) around every module whose PyTorch name is in `names`.
Fails for structs whose fields are not type parameters (use `@concrete`).
"""
function tapify(x, names, sink, key = "")
    y = if layer_params(x) !== nothing || x isa AbstractArray{<:Number} || x isa Function ||
           Functors.isleaf(x)
        x
    elseif x isa Chain
        layers = tapchildren(x.layers, names, sink, key)
        layers isa Tuple ? Chain(layers...) : Chain(layers)
    elseif x isa Union{Tuple, AbstractVector, NamedTuple}
        tapchildren(x, names, sink, key)
    else
        children, re = Functors.functor(x)
        re(tapchildren(children, names, sink, key))
    end
    return key in names && y !== nothing ? Tap(y, key, sink) : y
end

function tapchildren(t::Tuple, names, sink, key)
    return ntuple(i -> tapify(t[i], names, sink, join_key(key, i - 1)), length(t))
end
function tapchildren(v::AbstractVector, names, sink, key)
    return [tapify(c, names, sink, join_key(key, i - 1)) for (i, c) in enumerate(v)]
end
function tapchildren(nt::NamedTuple, names, sink, key)
    return NamedTuple{keys(nt)}(map(k -> tapify(nt[k], names, sink, join_key(key, k)), keys(nt)))
end

function parse_tap(key)
    m = match(r"^tap/(.*)#(\d+)(?:\.(\d+))?$", key)
    return m.captures[1], parse(Int, m.captures[2]),
    m.captures[3] === nothing ? nothing : parse(Int, m.captures[3])
end

"""
    shrink_channels(model, k; keep = ()) -> model

Rebuild the `Conv`, `BatchNorm` and `Dense` layers of `model` with every channel count
divided by `k`, like `shrunk_channels` in porting/common.py does for the PyTorch model. A count
in `keep` stays as is (the image channels and `num_classes`), every other count must be
divisible by `k`. Strides, padding, dilation, activations and `eps` are kept, and so is `groups` of a
depthwise convolution (`groups == in channels`); other `groups` stay as they are. The new
layers are zero-initialised, to be filled by `load_state!`. Fails on any other parameter.
"""
function shrink_channels(model, k::Integer; keep = ())
    function shrink(n)
        n in keep && return n
        n % k == 0 || error("channel count $n is not divisible by shrink = $k")
        return n ÷ k
    end
    isshrinkable(x) = x isa Union{Conv, BatchNorm, Dense}
    return fmap(model; exclude = x -> isshrinkable(x) || x isa AbstractArray{<:Number}) do l
        return l isa Conv ? shrink_conv(l, shrink) :
               l isa BatchNorm ? shrink_batchnorm(l, shrink) :
               l isa Dense ? shrink_dense(l, shrink) :
               error("cannot shrink a parameter array of size $(size(l))")
    end
end

function shrink_conv(l::Conv, shrink)
    (; weight, bias, groups) = l
    cin, cout = size(weight, ndims(weight) - 1) * groups, size(weight, ndims(weight))
    new_cin, new_cout = shrink(cin), shrink(cout)
    new_groups = groups > 1 && groups == cin ? new_cin : groups
    weight = zeros(
        eltype(weight), size(weight)[1:(end - 2)]..., new_cin ÷ new_groups, new_cout)
    bias = bias isa AbstractArray ? zeros(eltype(bias), new_cout) : bias
    return Conv(l.σ, weight, bias, l.stride, l.pad, l.dilation, new_groups, l.pad_mode)
end

function shrink_batchnorm(l::BatchNorm, shrink)
    new = BatchNorm(shrink(l.chs), l.λ; eps = l.ϵ, momentum = l.momentum,
        affine = l.affine, track_stats = l.track_stats)
    return Metalhead.ELTYPE_CONVERTERS[eltype(l.γ)](new)
end

function shrink_dense(l::Dense, shrink)
    out, in = size(l.weight)
    new = Dense(shrink(in) => shrink(out), l.σ; bias = l.bias isa AbstractArray)
    return Metalhead.ELTYPE_CONVERTERS[eltype(l.weight)](new)
end

"""
    fixture_model(dir; eltype = Float64)

The model of the fixture in `dir` with the weights of `model.safetensors`, in test mode. For
a fixture made from a `"shrink": k` spec, the model of `config.json` is built at full size,
shrunk with [`shrink_channels`](@ref) and then loaded.
"""
function fixture_model(dir; eltype = Float64)
    config = Metalhead.JSON.parsefile(joinpath(dir, "config.json"))
    haskey(config, "shrink") || return create_model(dir; pretrained = true, eltype)
    keep = (config["pretrained_cfg"]["input_size"][1], config["num_classes"])
    model = shrink_channels(create_model(dir; pretrained = false), config["shrink"]; keep)
    model = Metalhead.ELTYPE_CONVERTERS[eltype](model)
    spec = Metalhead.lookup(config["architecture"])
    Metalhead.load_state!(model, Checkpoint(joinpath(dir, "model.safetensors"));
        ignore = [Metalhead.DEFAULT_IGNORE; spec.ignore])
    return Flux.testmode!(model)
end

"""
    reference_model(file)

The model of the reference `file` of a language model, written by porting/dump_reference.py:
that of its repository at its commit. A reference of the first layers of a checkpoint
(`num_hidden_layers` in its metadata, from `dump_reference.py --layers`) gets the model of
`config.json` with those layers only, whose weights load strictly. The weights of each other
layer must then have the keys and shapes of the first of those layers of the same type in
`layer_types`, so that the whole checkpoint would load too.
"""
function reference_model(file)
    meta = SafeTensors.deserialize(file).metadata
    repo, revision = meta["repo"], meta["revision"]
    haskey(meta, "num_hidden_layers") ||
        return create_model(repo; pretrained = true, revision)
    k = parse(Int, meta["num_hidden_layers"])
    source = Metalhead.model_source(repo, revision, "")
    config = Metalhead.source_config(source)
    nested = haskey(config, "text_config")
    text = nested ? config["text_config"] : config
    n = text["num_hidden_layers"]
    types = get(text, "layer_types", nothing)
    first_layers = Dict{String, Any}("num_hidden_layers" => k)
    types === nothing || (first_layers["layer_types"] = types[1:k])
    kwargs = nested ? (; text_config = Metalhead.nt(merge(text, first_layers))) :
             Metalhead.nt(first_layers)
    model = create_model(repo; pretrained = false, revision, kwargs...)
    ckpt = Checkpoint(Metalhead.source_weights(source))
    prefix = last(Metalhead.Transformers.decoder(model)) * ".layers."
    function shapes(i)
        layer = "$prefix$i."
        return Dict(chopprefix(key, layer) => Metalhead.torch_shape(ckpt, key)
        for key in keys(ckpt) if startswith(key, layer))
    end
    for i in k:(n - 1)
        j = types === nothing ? 0 : findfirst(==(types[i + 1]), types) - 1
        j < k || error("layer $i is a $(types[i + 1]) layer, unlike the first $k")
        shapes(i) == shapes(j) ||
            error("the weights of layer $i differ in keys or shapes from those of layer $j")
    end
    other_layers = Regex("^" * replace(prefix, "." => "\\.") * "(" * join(k:(n - 1), "|") *
                         ")\\.")
    spec = Metalhead.lookup(only(config["architectures"]))
    Metalhead.load_state!(model, ckpt;
        ignore = [Metalhead.DEFAULT_IGNORE; spec.ignore; other_layers])
    return model
end

"""
    check_lm_reference(file; model = reference_model(file)) -> NamedTuple

Compare a causal language model with the reference outputs of its real checkpoint, written by
porting/dump_reference.py: the relative errors of the hidden states and of the logits of the
last position, and whether greedy decoding reproduces transformers' continuation of the first
prompt.
"""
function check_lm_reference(file; model = reference_model(file))
    meta = SafeTensors.deserialize(file).metadata
    ref = Checkpoint(file)
    ids = ref["input_ids"] .+ 1
    hidden = forward_features(model, ids)
    logits = forward_head(model, hidden[:, end, :])
    expected = vec(ref["generated_ids"]) .+ 1
    generated = greedy_decode(model, ids[:, 1], length(expected) - size(ids, 1))
    return (; repo = meta["repo"], hidden_states = relerr(hidden, ref["hidden_states"]),
        last_logits = relerr(logits, ref["last_logits"]), generated = generated == expected)
end

"""
    check_image_reference(file; model = reference_model(file)) -> NamedTuple

Compare a vision-language model with the reference outputs of its real checkpoint on a
prompt about an image ("image/..." keys, written by porting/dump_reference.py): the relative
errors of the hidden states and of the logits of the last position, and whether greedy
decoding reproduces transformers' continuation, from the same `pixel_values` and
`image_grid_thw`.
"""
function check_image_reference(file; model = reference_model(file))
    meta = SafeTensors.deserialize(file).metadata
    ref = Checkpoint(file)
    ids = ref["image/input_ids"] .+ 1
    images = (; pixel_values = Float32.(ref["image/pixel_values"]),
        image_grid_thw = ref["image/image_grid_thw"])
    hidden = forward_features(model, ids; images...)
    logits = forward_head(model, hidden[:, end, :])
    expected = vec(ref["image/generated_ids"]) .+ 1
    generated = greedy_decode(model, ids[:, 1], length(expected) - size(ids, 1); images...)
    return (; repo = meta["repo"],
        hidden_states = relerr(hidden, ref["image/hidden_states"]),
        last_logits = relerr(logits, ref["image/last_logits"]),
        generated = generated == expected)
end

"""
    check_sam3_reference(file) -> NamedTuple

Compare SAM 3 with the reference outputs of its real checkpoint, written by
porting/dump_reference.py, on the inputs that the processor of the repository
(`create_processor`) makes of the stored image, for a text prompt ("text/...") and a box
prompt ("box/...") that share its vision features: the largest relative errors of the
logits, boxes and presence logits, of the masks of the best queries and of the semantic
segmentation (every `stride` pixels), and of the scores and boxes of the processor's
`post_process_instance_segmentation`, and whether its numbers of objects and the areas of
their masks are those of transformers. The prompts are the strings of
porting/dump_reference.py, "cat" and, for the box prompt, the "visual" of a box without text,
which the processor encodes with the tokenizer of the repository (HuggingFaceTokenizers.jl
must be loaded), and the box of the box prompt is in pixels: its `input_ids`,
`attention_mask`, `input_boxes` and `input_boxes_labels` must be transformers'.
"""
function check_sam3_reference(file)
    meta = SafeTensors.deserialize(file).metadata
    ref = Checkpoint(file)
    T = Metalhead.Transformers
    model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
    proc = create_processor(meta["repo"]; revision = meta["revision"])
    images = proc(; images = Float32.(ref["image"]))
    vision = T.get_vision_features(model, images.pixel_values)
    s = parse(Int, meta["stride"])
    errs = Dict(k => 0.0 for k in (:logits, :boxes, :presence, :masks, :semantic, :post))
    record!(k, x, key) = errs[k] = max(errs[k], relerr(x, ref[key]))
    post_same = true
    for (name, text) in ("text" => "cat", "box" => nothing)
        boxes = haskey(ref, "$name/input_boxes") ?
                [collect(eachcol(ref["$name/boxes_xyxy"]))] : nothing
        inputs = proc(; images.original_sizes, text, input_boxes = boxes)
        post_same &= inputs.input_ids == ref["$name/input_ids"] .+ 1 &&
                     inputs.attention_mask == ref["$name/attention_mask"]
        if boxes !== nothing
            post_same &= inputs.input_boxes ≈ ref["$name/input_boxes"] &&
                         inputs.input_boxes_labels == ref["$name/input_boxes_labels"]
        end
        prompt = Base.structdiff(inputs, (; original_sizes = nothing, input_ids = nothing))
        y = model(vision, inputs.input_ids; prompt...)
        record!(:logits, y.pred_logits, "$name/pred_logits")
        record!(:boxes, y.pred_boxes, "$name/pred_boxes")
        record!(:presence, y.presence_logits, "$name/presence_logits")
        queries = vec(ref["$name/mask_queries"])
        record!(:masks, y.pred_masks[1:s:end, 1:s:end, queries, :], "$name/pred_masks")
        record!(:semantic, y.semantic_seg[1:s:end, 1:s:end, :, :], "$name/semantic_seg")
        post = only(T.post_process_instance_segmentation(proc, y; threshold = 0.5,
            mask_threshold = 0.5, target_sizes = images.original_sizes))
        post_same &= length(post.scores) == length(ref["$name/post/scores"]) &&
                     vec(sum(post.masks; dims = (1, 2))) == ref["$name/post/mask_areas"]
        if post_same
            record!(:post, post.scores, "$name/post/scores")
            record!(:post, post.boxes, "$name/post/boxes")
        end
    end
    return (; repo = meta["repo"], NamedTuple(errs)..., post_same)
end

"""
    check_siglip_reference(file) -> NamedTuple

Compare SigLIP or SigLIP 2 with the reference outputs of its real checkpoint, written by
porting/dump_reference.py, on the inputs that the processor of the repository
(`create_processor`) makes of the stored 8-bit image and of the texts of the metadata, which
it encodes with the tokenizer of the repository (HuggingFaceTokenizers.jl must be loaded):
whether those inputs are transformers' (its token ids, the stored pixels or patch features,
the grids and masks of the patches), and the largest relative errors of the logits, of the
normalized embeddings, of the outputs of the text and vision models (`pooler_output`) and
of their hidden states, every `stride` tokens.
"""
function check_siglip_reference(file)
    meta = SafeTensors.deserialize(file).metadata
    ref = Checkpoint(file)
    model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
    proc = create_processor(meta["repo"]; revision = meta["revision"])
    s = parse(Int, meta["stride"])
    inputs = proc(; images = ref["image"],
        text = Vector{String}(Metalhead.JSON.parse(meta["texts"])))
    pixels = ndims(inputs.pixel_values) == 4 ? inputs.pixel_values[1:s:end, 1:s:end, :, :] :
             inputs.pixel_values[1:s:end, 1:(s ÷ 2):end, :]
    same = inputs.input_ids == ref["input_ids"] .+ 1 && pixels == ref["pixel_values"] &&
           all(k -> !haskey(ref, string(k)) || inputs[k] == ref[string(k)],
        (:spatial_shapes, :pixel_attention_mask))
    y = model(inputs)
    err(k) = relerr(y[k], ref[string(k)])
    part(p, k) = relerr(k === :last_hidden_state ? y[p][k][:, 1:s:end, :] : y[p][k],
        ref["$p/$k"])
    return (; repo = meta["repo"], inputs = same, logits = max(err(:logits_per_image),
            err(:logits_per_text)),
        embeds = max(err(:text_embeds), err(:image_embeds)),
        pooled = max(part(:text_model_output, :pooler_output),
            part(:vision_model_output, :pooler_output)),
        hidden = max(part(:text_model_output, :last_hidden_state),
            part(:vision_model_output, :last_hidden_state)))
end

"""
    check_pipeline_fixture(dir; eltype = Float64, device = identity) -> NamedTuple

Compare a diffusers text-to-image pipeline written by porting/dump.py (a directory with a
model_index.json) with diffusers': the relative errors of the prompt embeddings of its token ids
(and of the negative prompt's, for classifier-free guidance), and of the denoised latents and
the images that the pipeline makes from diffusers' embeddings and initial noise. With an
"edit" in the spec, also those of the edit of its reference images (`edit_latents`,
`edit_image`), and of the reference images that the pipeline prepares (`reference_images`).
`device` (e.g. `gpu_device()`) moves the pipeline and its inputs.
"""
function check_pipeline_fixture(dir; eltype = Float64, device = identity)
    file = joinpath(dir, "acts.safetensors")
    acts = Checkpoint(file)
    spec = Metalhead.JSON.parse(SafeTensors.deserialize(file).metadata["spec"])
    pipe = device(create_model(dir; pretrained = true, eltype))
    input(key) = device(Base.eltype(acts[key]) <: AbstractFloat ? eltype.(acts[key]) :
                        acts[key])
    pipe isa Metalhead.Diffusers.ZImagePipeline &&
        return check_z_image_fixture(pipe, acts, spec, input)
    encode(prefix) = cpu(Metalhead.Diffusers.encode_prompt(pipe,
        input(prefix * "input_ids") .+ 1; attention_mask = input(prefix * "attention_mask"),
        text_encoder_out_layers = Tuple(spec["text_encoder_out_layers"])))
    guided = haskey(acts, "negative_prompt_embeds")
    kwargs = (; height = spec["height"], width = spec["width"],
        num_inference_steps = spec["num_inference_steps"],
        guidance_scale = spec["guidance_scale"], latents = input("latents"),
        negative_prompt_embeds = guided ? input("negative_prompt_embeds") : nothing)
    latents = cpu(pipe(input("prompt_embeds"); kwargs..., output_type = :latent))
    image = cpu(pipe(input("prompt_embeds"); kwargs...))
    edit = haskey(spec, "edit") ? check_pipeline_edit(pipe, acts, spec, kwargs, input) :
           (; edit_latents = 0.0, edit_image = 0.0, reference_images = 0.0)
    return (; prompt_embeds = relerr(encode(""), acts["prompt_embeds"]),
        negative_prompt_embeds = guided ?
                                 relerr(encode("negative_"), acts["negative_prompt_embeds"]) :
                                 0.0,
        latents = relerr(latents, acts["output_latents"]),
        image = relerr(image, acts["output"]), edit...)
end

# The values in [-1, 1] of the 8-bit values of an image that a pipeline prepares.
from_8bit(x) = 2 .* (Float32.(x) ./ 255) .- 1

function check_pipeline_edit(pipe, acts, spec, kwargs, input)
    edit = spec["edit"]
    n = count(k -> startswith(k, "edit/image/"), keys(acts))
    images = [acts["edit/image/$i"] for i in 1:n]
    f = 2 * Metalhead.Diffusers.vae_scale_factor(pipe)
    prepared = [relerr(Metalhead.Diffusers.reference_image(x, f),
                    from_8bit(acts["edit/prepared/$i"])) for (i, x) in enumerate(images)]
    hw = (; height = get(edit, "height", nothing), width = get(edit, "width", nothing))
    kwargs = merge(kwargs, (; image = images, hw..., latents = input("edit/latents")))
    latents = cpu(pipe(input("prompt_embeds"); kwargs..., output_type = :latent))
    image = cpu(pipe(input("prompt_embeds"); kwargs...))
    return (; edit_latents = relerr(latents, acts["edit/output_latents"]),
        edit_image = relerr(image, acts["edit/output"]), reference_images = maximum(prepared))
end

# The Z-Image pipeline takes the prompt embeddings as a list, "prompt_embeds/<i>", whose
# lengths differ, and has the classifier-free guidance options of the spec.
function check_z_image_fixture(pipe, acts, spec, input)
    embeds(key) = [input("$key/$j") for j in 1:count(startswith("$key/"), keys(acts))]
    joined(xs) = reduce(hcat, cpu.(xs))
    encode(prefix) = joined(Metalhead.Diffusers.encode_prompt(pipe,
        input(prefix * "input_ids") .+ 1; attention_mask = input(prefix * "attention_mask")))
    guided = haskey(acts, "negative_prompt_embeds/1")
    kwargs = (; height = spec["height"], width = spec["width"],
        num_inference_steps = spec["num_inference_steps"],
        guidance_scale = spec["guidance_scale"],
        cfg_normalization = get(spec, "cfg_normalization", false),
        cfg_truncation = get(spec, "cfg_truncation", 1.0), latents = input("latents"),
        negative_prompt_embeds = guided ? embeds("negative_prompt_embeds") : nothing)
    latents = cpu(pipe(embeds("prompt_embeds"); kwargs..., output_type = :latent))
    image = cpu(pipe(embeds("prompt_embeds"); kwargs...))
    return (; prompt_embeds = relerr(encode(""), joined(acts_list(acts, "prompt_embeds"))),
        negative_prompt_embeds = guided ?
                                 relerr(encode("negative_"),
            joined(acts_list(acts, "negative_prompt_embeds"))) : 0.0,
        latents = relerr(latents, acts["output_latents"]),
        image = relerr(image, acts["output"]))
end

"""
    check_diffusers_reference(file; eltype = Float32) -> NamedTuple

Compare a diffusers model or pipeline with the reference outputs of its real checkpoint,
written by porting/dump_reference.py: the relative errors of the latents and images of an
autoencoder, of the output of a transformer, or of the prompt embeddings, denoised latents and
image of a pipeline, which starts from its own prompt embeddings (and the same of an edit of
a reference image, `edit_...`, when the reference has one).
"""
function check_diffusers_reference(file; eltype = Float32)
    meta = SafeTensors.deserialize(file).metadata
    ref = Checkpoint(file)
    model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"], eltype)
    D = Metalhead.Diffusers
    class = meta["class"]
    # the difference of the parameter counts, when the reference has PyTorch's
    counts = haskey(meta, "num_params") ?
             (; num_params = abs(sum(length, Flux.trainables(model)) - parse(Int, meta["num_params"]))) :
             (;)
    if startswith(class, "Autoencoder")
        x = Base.eltype(ref["output"]).(reference_input(Tuple(Metalhead.JSON.parse(meta["input_shape"]))))
        latents = D.mode(D.encode(model, x))
        return (; repo = meta["repo"], latents = relerr(latents, ref["latents"]),
            output = relerr(D.decode(model, latents), ref["output"]), counts...)
    elseif class == "Flux2Transformer2DModel"
        inputs = [x isa AbstractArray{Float16} ? eltype.(x) : x
                  for x in (ref["inputs/$i"] for i in 1:5)]
        return (; repo = meta["repo"], output = relerr(model(inputs...), ref["output"]))
    elseif class == "ZImageTransformer2DModel"
        half(x) = x isa AbstractArray{Float16} ? eltype.(x) : x
        y = model(half(ref["inputs/1"]), ref["inputs/2"], half.(acts_list(ref, "inputs/3")))
        return (; repo = meta["repo"], output = relerr(y, ref["output"]), counts...)
    elseif class == "ZImagePipeline"
        return check_z_image_reference(model, ref, meta, eltype)
    end
    steps = parse(Int, meta["num_inference_steps"])
    # the edit of a reference image takes its size
    function generate(prefix; kwargs...)
        embeds = D.encode_prompt(model, ref[prefix * "input_ids"] .+ 1;
            attention_mask = ref[prefix * "attention_mask"])
        n = size(ref[prefix * "prompt_embeds"], 2)
        kwargs = (; kwargs..., num_inference_steps = steps, guidance_scale = 1.0,
            latents = eltype.(ref[prefix * "latents"]))
        latents = model(embeds; kwargs..., output_type = :latent)
        image = model(embeds; kwargs...)
        return (; prompt_embeds = relerr(embeds[:, 1:n, :], ref[prefix * "prompt_embeds"]),
            latents = relerr(latents, ref[prefix * "output_latents"]),
            image = relerr(image, ref[prefix * "output"]))
    end
    errs = generate(""; height = parse(Int, meta["height"]), width = parse(Int, meta["width"]))
    haskey(ref, "edit/output") || return (; repo = meta["repo"], errs...)
    edit = generate("edit/"; image = ref["edit/image"])
    return (; repo = meta["repo"], errs..., edit_prompt_embeds = edit.prompt_embeds,
        edit_latents = edit.latents, edit_image = edit.image)
end

# The Z-Image pipeline starts from the embeddings of its prompt, a list of one, of which the
# reference has the first positions.
function check_z_image_reference(pipe, ref, meta, eltype)
    embeds = Metalhead.Diffusers.encode_prompt(pipe, ref["input_ids"] .+ 1;
        attention_mask = ref["attention_mask"])
    n = size(ref["prompt_embeds"], 2)
    kwargs = (; height = parse(Int, meta["height"]), width = parse(Int, meta["width"]),
        num_inference_steps = parse(Int, meta["num_inference_steps"]),
        guidance_scale = parse(Float64, meta["guidance_scale"]),
        latents = eltype.(ref["latents"]))
    latents = pipe(embeds; kwargs..., output_type = :latent)
    image = pipe(embeds; kwargs...)
    return (; repo = meta["repo"], prompt_embeds = relerr(embeds[1][:, 1:n], ref["prompt_embeds"]),
        latents = relerr(latents, ref["output_latents"]), image = relerr(image, ref["output"]))
end

"""
    acts_list(acts, key) -> Vector

A list stored as "<key>/1", "<key>/2", ..., such as the captions of the Z-Image transformer,
whose lengths differ.
"""
acts_list(acts, key) = [acts["$key/$j"] for j in 1:count(startswith("$key/"), keys(acts))]

function greedy_decode(model, ids, n; kwargs...)
    for _ in 1:n
        ids = [ids; argmax(model(ids; kwargs...)[:, end])]
    end
    return ids
end

"""
    fixture_inputs(acts) -> (args, kwargs)

The inputs of the model of a fixture: "input" (token ids are made 1-based) or the positional
"inputs/1", "inputs/2", ..., a list being "inputs/<i>/1", "inputs/<i>/2", ... (see
[`acts_list`](@ref)), and the keyword arguments "kwargs/<name>", of which a tensor of no
dimensions is a scalar (a flag such as `interpolate_pos_encoding`).
"""
function fixture_inputs(acts)
    args = if haskey(acts, "input")
        x = acts["input"]
        (Base.eltype(x) <: Integer ? x .+ 1 : x,)  # 0-based token ids
    else
        n = maximum(k -> parse(Int, split(k, '/')[2]), filter(startswith("inputs/"), keys(acts)))
        Tuple(haskey(acts, "inputs/$i") ? acts["inputs/$i"] : acts_list(acts, "inputs/$i")
        for i in 1:n)
    end
    scalar(x) = ndims(x) == 0 ? x[] : x
    kwargs = (; (Symbol(chopprefix(k, "kwargs/")) => scalar(acts[k])
                 for k in sort!(filter(startswith("kwargs/"), collect(keys(acts)))))...)
    return args, kwargs
end

"""
    output_error(y, acts) -> Float64

The relative error of the output `y` of a fixture's model against its "output", or, for a
model that returns a NamedTuple (a vision encoder), the largest error of its fields against
"output/<name>", and of the fields of a field that is a NamedTuple against
"output/<name>/<field>".
"""
function output_error(y, acts, key = "output")
    y isa NamedTuple || return relerr(y, acts[key])
    return maximum(k -> output_error(y[k], acts, "$key/$k"), keys(y))
end

"""
    check_fixture(dir; rtol = 1e-8, model = fixture_model(dir)) -> NamedTuple

Load the tiny fixture in `dir` with [`fixture_model`](@ref) (in float64) and
compare its logits and features (float64, relative error `rtol`) and every recorded module output
with PyTorch's. Module outputs are stored in float16 or float32 and compared at a tolerance set
by their precision, and so are the `forward_intermediates` (see [`check_intermediates`](@ref)).
A fixture without features (a diffusion model) has `features = 0.0`. The output of a model
that returns a NamedTuple is compared field by field (see [`output_error`](@ref)).
Returns the relative errors, with the taps in PyTorch call order.
"""
function check_fixture(dir; rtol = 1e-8, model = fixture_model(dir))
    acts = Checkpoint(joinpath(dir, "acts.safetensors"))
    meta = SafeTensors.deserialize(joinpath(dir, "acts.safetensors")).metadata
    order = Metalhead.JSON.parse(meta["call_order"])
    args, kwargs = fixture_inputs(acts)
    output = output_error(model(args...; kwargs...), acts)
    features = haskey(acts, "features") ?
               relerr(forward_features(model, args...; kwargs...), acts["features"]) : 0.0
    tapkeys = filter(startswith("tap/"), collect(keys(acts)))
    names = Set(first(parse_tap(k)) for k in tapkeys)
    sink = Dict{String, Vector{Any}}()
    tapify(model, names, sink)(args...; kwargs...)
    taps = map(order) do k
        name, call, part = parse_tap("tap/" * k)
        ref = acts["tap/" * k]
        got = get(sink, name, Any[])
        y = call <= length(got) ? got[call] : nothing
        y = part === nothing || y === nothing ? y : y[part]
        err = y === nothing ? missing : size(y) == size(ref) ? relerr(y, ref) : Inf
        return (; name = k, err, tol = TAP_RTOL[eltype(ref)])
    end
    intermediates = check_intermediates(model, first(args), acts, meta; kwargs...)
    return (; output,
        features,
        taps,
        intermediates,
        pass = output <= rtol && features <= rtol &&
               all(t -> ismissing(t.err) || t.err <= t.tol, taps) &&
               all(t -> t.err <= t.tol, intermediates))
end

"""
    check_intermediates(model, x, acts, meta = Dict(); kwargs...) -> Vector{NamedTuple}

Compare `forward_intermediates(model, x; kwargs...)` (plain, with `norm = true` and the prefix
tokens of vision transformers) with the references "intermediates/<i>",
"intermediates_norm/<i>" and "intermediates_prefix/<i>" of a fixture, and its features with
"features". Without "intermediates_norm", the normed intermediates must equal the plain
references. `kwargs` are the model's keyword arguments (a language model's `attention_mask`).
"""
function check_intermediates(model, x, acts, meta = Dict(); kwargs...)
    n = count(startswith("intermediates/"), keys(acts))
    n == 0 && return NamedTuple[]
    errs = NamedTuple[]
    record!(name, y, ref, tol) = push!(errs,
        (; name, err = size(y) == size(ref) ? relerr(y, ref) : Inf, tol))
    tol = TAP_RTOL[eltype(acts["intermediates/1"])]
    if haskey(meta, "feature_info")
        ref = [Metalhead.nt(f) for f in Metalhead.JSON.parse(meta["feature_info"])]
        push!(errs, (; name = "feature_info", err = feature_info(model) == ref ? 0.0 : Inf,
            tol = 0.0))
    end
    features, plain = forward_intermediates(model, x; kwargs...)
    record!("features", features, acts["features"], 1e-8)
    _, normed = forward_intermediates(model, x; norm = true, kwargs...)
    length(plain) == length(normed) == n || return [(; name = "count", err = Inf, tol)]
    for i in 1:n
        record!("intermediates/$i", plain[i], acts["intermediates/$i"], tol)
        key = haskey(acts, "intermediates_norm/$i") ? "intermediates_norm/$i" :
              "intermediates/$i"
        record!("intermediates_norm/$i", normed[i], acts[key], tol)
    end
    take = unique([1, n])
    selected = forward_intermediates(model, x; indices = reverse(take),
        intermediates_only = true, kwargs...)
    early = forward_intermediates(model, x; indices = [1], stop_early = true,
        intermediates_only = true, kwargs...)
    for (i, y) in zip(take, selected)
        record!("selected/$i", y, acts["intermediates/$i"], tol)
    end
    record!("early/1", only(early), acts["intermediates/1"], tol)
    if haskey(acts, "intermediates_prefix/1")
        _, pairs = forward_intermediates(model, x; return_prefix_tokens = true)
        for i in 1:n
            record!("intermediates_prefix/$i", last(pairs[i]),
                acts["intermediates_prefix/$i"], tol)
        end
    end
    return errs
end

end # module
