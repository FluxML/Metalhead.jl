# Ported from diffusers.pipelines.flux2.pipeline_flux2_klein (diffusers 0.42.0, Apache-2.0,
# The HuggingFace Team): text-to-image generation, without the conditioning images of image
# editing, LoRA and callbacks. Validated against test/fixtures/flux2_klein_tiny and
# test/references/black-forest-labs--FLUX.2-klein-4B.

"""
    Flux2KleinPipeline

diffusers' text-to-image pipeline of FLUX.2 [klein], built by
[`create_model`](@ref Metalhead.create_model) from a repository such as
`"black-forest-labs/FLUX.2-klein-4B"`. Its components are
`text_encoder`, a Qwen3 language model whose hidden states condition the image; `transformer`,
a [`Flux2Transformer2DModel`](@ref Metalhead.Diffusers.flux2_transformer_2d_model) that
predicts the flow of the latents; `vae`, the
[`AutoencoderKLFlux2`](@ref Metalhead.Diffusers.autoencoder_kl_flux2) that decodes them;
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
# image tokens, (0, h, w, 0) with w running fastest, as integer arrays of size (4, seq_len).
text_ids(n) = vcat(zeros(Int, 3, n), reshape(collect(0:(n - 1)), 1, n))
latent_ids(w, h) = reduce(hcat, vec([[0, j, i, 0] for i in 0:(w - 1), j in 0:(h - 1)]))

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
    (pipeline::Flux2KleinPipeline)(prompt; height = 1024, width = 1024, num_inference_steps = 50,
                                   guidance_scale = 4.0, max_sequence_length = 512,
                                   latents = nothing, rng = Random.default_rng(), ...)
    (pipeline::Flux2KleinPipeline)(prompt_embeds::AbstractArray{<:Real, 3}; ...)

Generate an image for each prompt (a string or a vector of strings, which the pipeline's
tokenizer encodes, see [`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt)), or for
the prompt embeddings of
[`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt), of size
`(joint_attention_dim, seq_len, batch)`. Returns an array of size `(width, height, 3, batch)`
with values in [0, 1]: `colorview(RGB, permutedims(image[:, :, :, 1], (3, 2, 1)))` makes an
image of the first one with ImageCore.jl.

Keyword arguments, with the defaults of diffusers:

  - `height`, `width`: the size of the image, rounded down to a multiple of 16.
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
        negative_prompt_embeds = nothing, height::Integer = 1024, width::Integer = 1024,
        num_inference_steps::Integer = 50, sigmas = nothing, guidance_scale = 4.0,
        latents = nothing, rng::AbstractRNG = Random.default_rng(),
        output_type::Symbol = :image)
    output_type in (:image, :latent) ||
        error("`output_type` must be :image or :latent, not $(repr(output_type))")
    guided = guidance_scale > 1 && !p.is_distilled
    guided && negative_prompt_embeds === nothing &&
        error("classifier-free guidance (`guidance_scale > 1`) needs `negative_prompt_embeds`, the embeddings of the empty prompt")
    f = 2 * vae_scale_factor(p)
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
    img_ids = latent_ids(shape[1], shape[2])
    txt_ids = text_ids(size(prompt_embeds, 2))
    sigmas = something(sigmas, range(1, 1 / num_inference_steps; length = num_inference_steps))
    mu = compute_empirical_mu(size(x, 2), num_inference_steps)
    timesteps, sigmas = set_timesteps(p.scheduler, num_inference_steps; sigmas, mu)
    for (i, t) in enumerate(timesteps)
        timestep = fill(T(t), b) ./ T(1000)
        v = p.transformer(x, prompt_embeds, timestep, img_ids, txt_ids)
        if guided
            v_neg = p.transformer(x, negative_prompt_embeds, timestep, img_ids,
                text_ids(size(negative_prompt_embeds, 2)))
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
