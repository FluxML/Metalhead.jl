# Ported from diffusers.pipelines.flux2.pipeline_flux2_klein and the preprocessing of its
# reference images in .image_processor (`Flux2ImageProcessor`) (diffusers 0.42.1, Apache-2.0,
# The HuggingFace Team): text-to-image generation and image editing with reference images,
# without `num_images_per_prompt`, LoRA and callbacks. Validated against
# test/fixtures/flux2_klein_tiny, test/fixtures/flux2_klein_cfg and
# test/references/black-forest-labs--FLUX.2-klein-4B.

"""
    Flux2KleinPipeline

diffusers' pipeline of FLUX.2 [klein], which generates images from text, or edits images
(reference images, `image`) as the text says, built by
[`create_model`](@ref Metalhead.create_model) from a repository such as
`"black-forest-labs/FLUX.2-klein-4B"`. Its components are
`text_encoder`, a Qwen3 language model whose hidden states condition the image; `transformer`,
a [`Flux2Transformer2DModel`](@ref Metalhead.Diffusers.flux2_transformer_2d_model) that
predicts the flow of the latents; `vae`, the
[`AutoencoderKLFlux2`](@ref Metalhead.Diffusers.autoencoder_kl_flux2) that decodes them (and
encodes the reference images);
`scheduler`, the [`FlowMatchEulerDiscreteScheduler`](@ref Metalhead.Diffusers.FlowMatchEulerDiscreteScheduler);
and `tokenizer`, the tokenizer of the text encoder: HuggingFaceTokenizers.jl's `Tokenizer`
of the `tokenizer/` subfolder. Without `using HuggingFaceTokenizers` before `create_model`,
it is a reference to the tokenizer.json, which the pipeline loads once HuggingFaceTokenizers
is loaded, without loading the pipeline again; it is `nothing` for a pipeline saved without
a tokenizer, which takes prompt embeddings only. `is_distilled` checkpoints (FLUX.2 [klein]
4B, 9B) sample in a few steps without classifier-free guidance; the base ones (FLUX.2
[klein] base 4B, 9B) take about 50 steps and a `guidance_scale` of 4.

`gpu(pipeline)` and `bf16(pipeline)` move and convert every component but the tokenizer.

```julia
using Metalhead, HuggingFaceTokenizers
pipe = create_model("black-forest-labs/FLUX.2-klein-4B"; eltype = BFloat16)
image = pipe("A cat holding a sign that says hello world"; num_inference_steps = 4)
edited = pipe("Make the sign say goodbye"; image = image[:, :, :, 1], num_inference_steps = 4)
```
"""
@concrete struct Flux2KleinPipeline
    scheduler
    vae
    text_encoder
    tokenizer
    transformer
    is_distilled
end
Flux.@layer Flux2KleinPipeline

function flux2_klein_pipeline(; scheduler = nothing, vae = nothing, text_encoder = nothing,
        tokenizer = nothing, transformer = nothing, is_distilled = false)
    any(isnothing, (scheduler, vae, text_encoder, transformer)) &&
        error("a Flux2KleinPipeline is built from a repository, e.g. `create_model(\"black-forest-labs/FLUX.2-klein-4B\")`")
    return Flux2KleinPipeline(scheduler, vae, text_encoder, tokenizer, transformer,
        is_distilled)
end

register!(ArchSpec(DIFFUSERS_PIPELINE, "Flux2KleinPipeline", flux2_klein_pipeline;
    family = "flux2",
    references = ["black-forest-labs/FLUX.2-klein-4B" => "e7b7dc27f91deacad38e78976d1f2b499d76a294"]))

vae_scale_factor(p::Flux2KleinPipeline) = 2^(length(p.vae.encoder.down_blocks) - 1)

"""
    encode_prompt(pipeline, prompt; max_sequence_length = 512,
                  text_encoder_out_layers = (9, 18, 27))
    encode_prompt(pipeline, input_ids; attention_mask = nothing,
                  text_encoder_out_layers = (9, 18, 27))

The embeddings of a prompt (a string) or of a vector of prompts, as diffusers'
`encode_prompt` makes them, of size
`(length(text_encoder_out_layers) * hidden_size, max_sequence_length, batch)`. Each prompt
is formatted with the chat template of the Qwen3 text encoder (a user message, with thinking
disabled and a generation prompt), tokenized with `pipeline.tokenizer` (which needs `using
HuggingFaceTokenizers`), truncated to `max_sequence_length` tokens and padded on the right
to them. The embeddings are the hidden states of the layers
`text_encoder_out_layers` of the text encoder (transformers' `output_hidden_states`, of
which that of the last layer is normed), concatenated along the features.

The second method takes the prompts tokenized: `input_ids`, 1-based token indices of size
`(seq_len, batch)`, padded on the right as `attention_mask` says, and moved to the device of
the text encoder.
"""
function encode_prompt(p::Flux2KleinPipeline,
        prompt::Union{AbstractString, AbstractVector{<:AbstractString}};
        max_sequence_length::Integer = 512, text_encoder_out_layers = (9, 18, 27))
    tok = resolve_tokenizer(p.tokenizer)
    if tok === nothing
        p.tokenizer isa DeferredTokenizer &&
            error("the tokenizer of the pipeline needs HuggingFaceTokenizers.jl to encode prompts: run `using HuggingFaceTokenizers`, and call the pipeline again, which needn't be loaded again; or pass prompt embeddings, or the token indices of the prompts to `encode_prompt`")
        error("the pipeline has no tokenizer to encode prompts, as its repository has no tokenizer/tokenizer.json: pass prompt embeddings, or the token indices of the prompts to `encode_prompt`")
    end
    inputs = qwen3_prompt_inputs(tok, prompt; max_sequence_length)
    return encode_prompt(p, inputs.input_ids; inputs.attention_mask,
        text_encoder_out_layers)
end

function encode_prompt(p::Flux2KleinPipeline, input_ids::AbstractMatrix{<:Integer};
        attention_mask = nothing, text_encoder_out_layers = (9, 18, 27))
    layers = collect(Int, text_encoder_out_layers)
    input_ids = Flux.get_device(p.text_encoder)(input_ids)
    hidden = forward_intermediates(p.text_encoder, input_ids; indices = layers,
        attention_mask, stop_early = true, intermediates_only = true)
    sorted = sort!(unique(layers))
    model = p.text_encoder.model
    state(l) = (h = hidden[searchsortedfirst(sorted, l)];
                l == length(model.layers) ? model.norm(h) : h)
    return reduce(vcat, [state(l) for l in layers])
end

# The token indices and attention mask `(max_sequence_length, batch)` of the prompts, as
# diffusers' `_get_qwen3_prompt_embeds` tokenizes them (`padding = "max_length"`,
# `truncation = True`).
function qwen3_prompt_inputs(tokenizer, prompt; max_sequence_length::Integer = 512)
    prompts = prompt isa AbstractString ? [prompt] : collect(prompt)
    ids = [first(encode_ids(tokenizer, chat_prompt(p)), max_sequence_length)
           for p in prompts]
    return pad_ids(ids, pad_token_id(tokenizer); max_length = max_sequence_length)
end

# The text that Qwen3's chat template makes of a user message, with thinking disabled and a
# generation prompt, as diffusers' FLUX.2 [klein] pipeline formats each prompt.
chat_prompt(prompt) = "<|im_start|>user\n$(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# The (t, h, w, l) coordinates of the text tokens, (0, 0, 0, l), and of a `w × h` grid of
# image tokens, (t, h, w, 0) with w running fastest, as integer arrays of size (4, seq_len).
# The generated image has t = 0, and its reference images t = 10, 20, ...
text_ids(n) = vcat(zeros(Int, 3, n), reshape(collect(0:(n - 1)), 1, n))
function latent_ids(w, h, t = 0)
    return reduce(hcat, vec([[t, j, i, 0] for i in 0:(w - 1), j in 0:(h - 1)]))
end

# A (W, H, C, B) feature map as (C, W * H, B) tokens, and back.
pack_latents(x) = permutedims(reshape(x, :, size(x, 3), size(x, 4)), (2, 1, 3))
unpack_latents(x, w, h) = reshape(permutedims(x, (2, 1, 3)), w, h, size(x, 1), size(x, 3))

# The 2 × 2 patches of a (W, H, C, B) feature map as the 4C channels of a (W/2, H/2) one, in
# diffusers' order of the channels, and back.
function patchify_latents(x)
    w, h, c, b = size(x)
    y = permutedims(reshape(x, 2, w ÷ 2, 2, h ÷ 2, c, b), (2, 4, 1, 3, 5, 6))
    return reshape(y, w ÷ 2, h ÷ 2, 4c, b)
end
function unpatchify_latents(x)
    w, h, c, b = size(x)
    y = permutedims(reshape(x, w, h, 2, 2, c ÷ 4, b), (3, 1, 4, 2, 5, 6))
    return reshape(y, 2w, 2h, c ÷ 4, b)
end

# The mean and the standard deviation of the patchified latents, of the `bn` of the autoencoder.
function latent_stats(vae)
    bn = vae.bn
    return reshape(bn.μ, 1, 1, :, 1), reshape(sqrt.(bn.σ² .+ bn.ϵ), 1, 1, :, 1)
end

# diffusers' `Flux2ImageProcessor.check_image_input`.
function check_reference_image(w, h)
    min(w, h) >= 64 ||
        throw(ArgumentError("a reference image of $w × $h pixels is too small: both sides must be at least 64 pixels"))
    max(w / h, h / w) <= 8 ||
        throw(ArgumentError("a reference image of $w × $h pixels has an aspect ratio over 8:1"))
    return nothing
end

# An image as the 8-bit PIL image that diffusers takes: reals from 0 to 1 are times 255,
# rounded, as diffusers' `numpy_to_pil` makes PIL images of the outputs of its pipelines.
function pil_8bit(image)
    x = image_array(image)
    if eltype(x) <: Integer
        all(v -> 0 <= v <= 255, x) ||
            throw(ArgumentError("an image of integers has values from 0 to 255"))
        return UInt8.(x)
    end
    all(v -> 0 <= v <= 1, x) ||
        throw(ArgumentError("an image of real numbers has values from 0 to 1"))
    return round.(UInt8, float.(x) .* 255)
end

# Pillow's resize, which resizes an image with alpha (LA or RGBA) with its colours
# premultiplied by the alpha (its modes La and RGBa), except with the nearest filter.
function pil_resize_alpha(x, (w, h), filter)
    c = size(x, 3)
    (c in (2, 4) && filter !== :nearest) || return pil_resize(x, (w, h), filter)
    alpha = x[:, :, c:c]
    premultiplied = pil_resize(cat(muldiv255.(x[:, :, 1:(c - 1)], alpha), alpha; dims = 3),
        (w, h), filter)
    alpha = premultiplied[:, :, c:c]
    return cat(unpremultiply.(premultiplied[:, :, 1:(c - 1)], alpha), alpha; dims = 3)
end
muldiv255(v, a) = (t = Int(v) * Int(a) + 128; UInt8(((t >> 8) + t) >> 8))
unpremultiply(v, a) = a == 0 || a == 255 ? v : UInt8(min(255 * Int(v) ÷ Int(a), 255))

# A reference image as diffusers' FLUX.2 [klein] pipeline prepares it, of size
# (width, height, 3) with values in [-1, 1]: an image of more than `max_area` pixels resized
# to about that area with Pillow's Lanczos filter, its sides cropped about the centre to
# multiples of `multiple_of`, then converted to RGB.
function reference_image(image, multiple_of::Integer; max_area::Integer = 1024^2)
    x = pil_8bit(image)
    w, h = size(x, 1), size(x, 2)
    check_reference_image(w, h)
    if w * h > max_area
        scale = sqrt(max_area / (w * h))
        w, h = trunc(Int, w * scale), trunc(Int, h * scale)
        x = pil_resize_alpha(x, (w, h), :lanczos)
    end
    cw, ch = w ÷ multiple_of * multiple_of, h ÷ multiple_of * multiple_of
    left, top = (w - cw) ÷ 2, (h - ch) ÷ 2
    x = convert_to_rgb(x[(left + 1):(left + cw), (top + 1):(top + ch), :])
    return 2 .* (Float32.(x) ./ 255) .- 1
end

# The tokens `(channels, seq_len, 1)` and position ids of the reference images, as diffusers'
# `prepare_image_latents` makes them: the mean of the posterior of each image, patchified
# and normalised by the statistics of the autoencoder, its tokens after those of the images
# before it.
function reference_latents(p::Flux2KleinPipeline, images)
    weight = p.vae.encoder.conv_in.weight
    mean, std = latent_stats(p.vae)
    latents = map(images) do x
        z = mode(encode(p.vae, to_device_of(weight, reshape(x, size(x)..., 1))))
        z = patchify_latents(z)
        return convert_like(z, (z .- mean) ./ std)
    end
    ids = reduce(hcat, [latent_ids(size(z, 1), size(z, 2), 10i)
                        for (i, z) in enumerate(latents)])
    return reduce(hcat, pack_latents.(latents)), ids
end

"""
    compute_empirical_mu(image_seq_len, num_steps)

The shift of the noise levels of FLUX.2 for `image_seq_len` image tokens and `num_steps` steps,
fitted by Black Forest Labs.
"""
function compute_empirical_mu(image_seq_len::Integer, num_steps::Integer)
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    image_seq_len > 4300 && return a2 * image_seq_len + b2
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return a * num_steps + b
end

"""
    (pipeline::Flux2KleinPipeline)(prompt; image = nothing, height = nothing, width = nothing,
                                   num_inference_steps = 50, guidance_scale = 4.0,
                                   max_sequence_length = 512, latents = nothing,
                                   rng = Random.default_rng(), ...)
    (pipeline::Flux2KleinPipeline)(prompt_embeds::AbstractArray{<:Real, 3}; ...)

Generate an image for each prompt (a string or a vector of strings, which the pipeline's
tokenizer encodes, see [`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt)), or for
the prompt embeddings of
[`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt), of size
`(joint_attention_dim, seq_len, batch)`; with reference images `image`, edit them as each
prompt says. Returns an array of size `(width, height, 3, batch)` with values in [0, 1]:
[`Metalhead.image_matrix`](@ref) makes images of it with ImageCore.jl.

Keyword arguments, with the defaults of diffusers:

  - `image`: the reference images, one image or a vector of them (or an array of size
    `(width, height, channels, n)`), each an array of size `(width, height, channels)` of
    8-bit values or of reals from 0 to 1, or a matrix of colorants with ImageCore (see
    [`Metalhead.image_array`](@ref)). Every prompt takes all of them. As diffusers prepares
    PIL images, each must have sides of at least 64 pixels and an aspect ratio of at most
    8:1; one of more than 1024 × 1024 pixels is resized to about that area with Pillow's
    Lanczos filter, on 8-bit values (reals are rounded to them); its sides are cropped about
    the centre to multiples of 16, and it is converted to RGB. The autoencoder encodes each,
    and the transformer attends to their tokens along with those of the image it generates.
  - `height`, `width`: the size of the image, rounded down to a multiple of 16; by default
    the size of the first reference image, once cropped, or else 1024.
  - `num_inference_steps`: 4 suit the distilled checkpoints, 50 the base ones.
  - `sigmas`: the noise levels of the steps, before the shift of FLUX.2
    (by default evenly spaced from 1 to `1 / num_inference_steps`).
  - `guidance_scale`: the scale of classifier-free guidance, for checkpoints that are not
    distilled, whose negative prompt is the empty one (the embeddings `negative_prompt_embeds`
    with prompt embeddings).
  - `latents`: the initial noise, of size `(width ÷ 16, height ÷ 16, in_channels, batch)`, drawn
    from `rng` on the CPU by default.
  - `output_type`: `:image`, or `:latent` for the denoised latents before the autoencoder.
  - `max_sequence_length`, `text_encoder_out_layers`: as in
    [`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt).
"""
function (p::Flux2KleinPipeline)(prompt::Union{AbstractString, AbstractVector{<:AbstractString}};
        max_sequence_length::Integer = 512, text_encoder_out_layers = (9, 18, 27),
        guidance_scale = 4.0, kwargs...)
    prompts = prompt isa AbstractString ? [prompt] : collect(prompt)
    embed(texts) = encode_prompt(p, texts; max_sequence_length, text_encoder_out_layers)
    negative = guidance_scale > 1 && !p.is_distilled ? embed(fill("", length(prompts))) :
               nothing
    return p(embed(prompts); negative_prompt_embeds = negative, guidance_scale, kwargs...)
end

function (p::Flux2KleinPipeline)(prompt_embeds::AbstractArray{<:Real, 3};
        image = nothing, negative_prompt_embeds = nothing,
        height::Union{Nothing, Integer} = nothing, width::Union{Nothing, Integer} = nothing,
        num_inference_steps::Integer = 50, sigmas = nothing, guidance_scale = 4.0,
        latents = nothing, rng::AbstractRNG = Random.default_rng(),
        output_type::Symbol = :image)
    output_type in (:image, :latent) ||
        error("`output_type` must be :image or :latent, not $(repr(output_type))")
    guided = guidance_scale > 1 && !p.is_distilled
    guided && negative_prompt_embeds === nothing &&
        error("classifier-free guidance (`guidance_scale > 1`) needs `negative_prompt_embeds`, the embeddings of the empty prompt")
    f = 2 * vae_scale_factor(p)
    references = image === nothing ? nothing :
                 [reference_image(x, f) for x in image_batch(image)]
    if references !== nothing
        isempty(references) && throw(ArgumentError("`image` has no images"))
        height = something(height, size(first(references), 2))
        width = something(width, size(first(references), 1))
    end
    height = something(height, 128 * vae_scale_factor(p))
    width = something(width, 128 * vae_scale_factor(p))
    (height % f == 0 && width % f == 0) ||
        @warn "`height` and `width` have to be divisible by $f but are $height and $width; they are rounded down"
    b = size(prompt_embeds, 3)
    channels = size(p.transformer.x_embedder.weight, 2)
    shape = (width ÷ f, height ÷ f, channels, b)
    latents = latents === nothing ? randn(rng, Float32, shape) : latents
    size(latents) == shape ||
        error("`latents` must have size $shape for these `height`, `width` and prompts")
    x = pack_latents(to_device_of(prompt_embeds, latents))
    T = eltype(x)
    n = size(x, 2)
    ref_tokens, img_ids = if references === nothing
        nothing, latent_ids(shape[1], shape[2])
    else
        tokens, ref_ids = reference_latents(p, references)
        repeat(convert_like(x, tokens), 1, 1, b),
        hcat(latent_ids(shape[1], shape[2]), ref_ids)
    end
    # the reference tokens follow those of the image, and their flows are dropped
    function flow(x, embeds, timestep)
        input = ref_tokens === nothing ? x : cat(x, ref_tokens; dims = 2)
        v = p.transformer(input, embeds, timestep, img_ids, text_ids(size(embeds, 2)))
        return ref_tokens === nothing ? v : v[:, 1:n, :]
    end
    sigmas = something(sigmas, range(1, 1 / num_inference_steps; length = num_inference_steps))
    mu = compute_empirical_mu(n, num_inference_steps)
    timesteps, sigmas = set_timesteps(p.scheduler, num_inference_steps; sigmas, mu)
    for (i, t) in enumerate(timesteps)
        timestep = fill(T(t), b) ./ T(1000)
        v = flow(x, prompt_embeds, timestep)
        if guided
            v_neg = flow(x, negative_prompt_embeds, timestep)
            v = v_neg .+ T(guidance_scale) .* (v .- v_neg)
        end
        x = step(p.scheduler, v, sigmas[i], sigmas[i + 1], x)
    end
    # Float32 under `bf16`, which keeps the statistics of a BatchNorm
    mean, std = latent_stats(p.vae)
    z = convert_like(x, unpack_latents(x, shape[1], shape[2]) .* std .+ mean)
    z = unpatchify_latents(z)
    output_type === :latent && return z
    return clamp.(decode(p.vae, z) ./ 2 .+ T(0.5), 0, 1)
end
