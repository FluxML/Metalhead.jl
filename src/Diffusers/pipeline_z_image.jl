# Ported from diffusers.pipelines.z_image.pipeline_z_image (diffusers 0.42.0, Apache-2.0,
# Alibaba Z-Image Team and The HuggingFace Team): text-to-image generation, without several
# images per prompt, LoRA and callbacks. Validated against test/fixtures/z_image_* and
# test/references/Tongyi-MAI--Z-Image-Turbo.

"""
    ZImagePipeline

diffusers' text-to-image pipeline of Z-Image, built by
[`create_model`](@ref Metalhead.create_model) from a repository such as
`"Tongyi-MAI/Z-Image-Turbo"`. Its components are `text_encoder`, a Qwen3 decoder (a
`LlamaModel`, transformers' `Qwen3Model`) whose hidden states condition the image;
`transformer`, a
[`ZImageTransformer2DModel`](@ref Metalhead.Diffusers.ZImageTransformer2DModel) that
predicts the flow of the latents; `vae`, the
[`AutoencoderKL`](@ref Metalhead.Diffusers.autoencoder_kl) that decodes them; `scheduler`,
the
[`FlowMatchEulerDiscreteScheduler`](@ref Metalhead.Diffusers.FlowMatchEulerDiscreteScheduler);
and `tokenizer`, the tokenizer of the text encoder, as for
[`Flux2KleinPipeline`](@ref Metalhead.Diffusers.Flux2KleinPipeline):
HuggingFaceTokenizers.jl's
`Tokenizer` of the `tokenizer/` subfolder, a reference to it that the pipeline loads once
HuggingFaceTokenizers is loaded, or `nothing` without one.

The defaults of the pipeline are those of diffusers, for Z-Image: 50 steps with a
`guidance_scale` of 5. Z-Image-Turbo is distilled to sample in 9 steps without guidance
(`guidance_scale = 0`).

`gpu(pipeline)` and `bf16(pipeline)` move and convert every component but the tokenizer.

```julia
using Metalhead, Flux, HuggingFaceTokenizers
pipe = create_model("Tongyi-MAI/Z-Image-Turbo"; eltype = BFloat16)
image = pipe("A cat holding a sign that says hello world"; num_inference_steps = 9,
             guidance_scale = 0)
```
"""
@concrete struct ZImagePipeline
    scheduler
    vae
    text_encoder
    tokenizer
    transformer
end
Flux.@layer ZImagePipeline

function z_image_pipeline(; scheduler = nothing, vae = nothing, text_encoder = nothing,
        tokenizer = nothing, transformer = nothing)
    any(isnothing, (scheduler, vae, text_encoder, transformer)) &&
        error("a ZImagePipeline is built from a repository, e.g. `create_model(\"Tongyi-MAI/Z-Image-Turbo\")`")
    return ZImagePipeline(scheduler, vae, text_encoder, tokenizer, transformer)
end

register!(ArchSpec(DIFFUSERS_PIPELINE, "ZImagePipeline", z_image_pipeline;
    family = "z_image",
    references = ["Tongyi-MAI/Z-Image-Turbo" => "f332072aa78be7aecdf3ee76d5c247082da564a6"]))

vae_scale_factor(p::ZImagePipeline) = 2^(length(p.vae.encoder.down_blocks) - 1)

"""
    encode_prompt(pipeline::ZImagePipeline, prompt; max_sequence_length = 512)
    encode_prompt(pipeline::ZImagePipeline, input_ids; attention_mask = nothing)

The embeddings of a prompt (a string) or of a vector of prompts, as diffusers'
`ZImagePipeline.encode_prompt` makes them: a vector with a matrix of size
`(hidden_size, length)` for each prompt, the hidden states of its tokens in the
second-to-last layer of the text encoder (transformers' `output_hidden_states[-2]`), without
the padding. Each prompt is formatted with the chat template of the Qwen3 text encoder (a
user message and a generation prompt, with thinking enabled), tokenized with
`pipeline.tokenizer` (which needs `using HuggingFaceTokenizers`) and truncated to
`max_sequence_length` tokens.

The second method takes the prompts tokenized: `input_ids`, 1-based token indices of size
`(seq_len, batch)`, padded on the right as `attention_mask` says.
"""
function encode_prompt(p::ZImagePipeline,
        prompt::Union{AbstractString, AbstractVector{<:AbstractString}};
        max_sequence_length::Integer = 512)
    inputs = z_image_prompt_inputs(prompt_tokenizer(p.tokenizer), prompt;
        max_sequence_length)
    return encode_prompt(p, inputs.input_ids; inputs.attention_mask)
end

function encode_prompt(p::ZImagePipeline, input_ids::AbstractMatrix{<:Integer};
        attention_mask = nothing)
    model = p.text_encoder
    x, position_embeddings, kv_lengths = embed(model, Flux.get_device(model)(input_ids),
        attention_mask)
    for layer in model.layers[1:(end - 1)]
        x = layer(x, position_embeddings, kv_lengths)
    end
    lengths = something(kv_lengths, fill(size(input_ids, 1), size(input_ids, 2)))
    return [x[:, 1:n, i] for (i, n) in enumerate(lengths)]
end

# The token indices and attention mask `(max_sequence_length, batch)` of the prompts, as
# diffusers' `_encode_prompt` tokenizes them (`padding = "max_length"`,
# `truncation = True`).
function z_image_prompt_inputs(tokenizer, prompt; max_sequence_length::Integer = 512)
    prompts = prompt isa AbstractString ? [prompt] : collect(prompt)
    ids = [first(encode_ids(tokenizer, z_image_chat_prompt(p)), max_sequence_length)
           for p in prompts]
    return pad_ids(ids, pad_token_id(tokenizer); max_length = max_sequence_length)
end

# The text that Qwen3's chat template makes of a user message, with a generation prompt and
# thinking enabled (no empty thinking block), as diffusers' Z-Image pipeline formats each
# prompt.
function z_image_chat_prompt(prompt)
    return "<|im_start|>user\n$(prompt)<|im_end|>\n<|im_start|>assistant\n"
end

"""
    (pipeline::ZImagePipeline)(prompt; negative_prompt = nothing, height = 1024,
                               width = 1024, num_inference_steps = 50, guidance_scale = 5.0,
                               max_sequence_length = 512, latents = nothing,
                               rng = Random.default_rng(), ...)
    (pipeline::ZImagePipeline)(prompt_embeds::AbstractVector{<:AbstractMatrix}; ...)

Generate an image for each prompt (a string or a vector of strings, which the pipeline's
tokenizer encodes, see [`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt)), or for
the prompt embeddings of `encode_prompt`, a vector of matrices of size
`(cap_feat_dim, length)`. Returns an array of size `(width, height, 3, batch)` with values
in [0, 1].

Keyword arguments, with the defaults of diffusers:

  - `height`, `width`: the size of the image, a multiple of 16.
  - `num_inference_steps`: 9 suit Z-Image-Turbo, 50 Z-Image.
  - `sigmas`: the noise levels of the steps, before the shift of the scheduler (by default
    evenly spaced from 1 to `1 / num_inference_steps`).
  - `guidance_scale`: the scale of classifier-free guidance when positive: the prediction is
    `pos + guidance_scale * (pos - neg)`, of the prompt and of the negative prompt
    (`negative_prompt`, by default the empty one; the embeddings `negative_prompt_embeds`
    with prompt embeddings). 0 for Z-Image-Turbo.
  - `cfg_normalization`: when positive, the guided prediction of each image is scaled down
    to at most `cfg_normalization` times the norm of the prediction of its prompt.
  - `cfg_truncation`: guidance stops at the steps whose timestep `1 - σ` exceeds it.
  - `latents`: the initial noise, of size `(width ÷ 8, height ÷ 8, in_channels, batch)`,
    drawn from `rng` on the CPU by default. The latents are denoised in `Float32` (or
    `Float64`).
  - `output_type`: `:image`, or `:latent` for the denoised latents before the autoencoder.
  - `max_sequence_length`: as in [`encode_prompt`](@ref Metalhead.Diffusers.encode_prompt).
"""
function (p::ZImagePipeline)(
        prompt::Union{AbstractString, AbstractVector{<:AbstractString}};
        negative_prompt = nothing, max_sequence_length::Integer = 512, guidance_scale = 5.0,
        kwargs...)
    prompts = prompt isa AbstractString ? [prompt] : collect(prompt)
    encode(texts) = encode_prompt(p, texts; max_sequence_length)
    negative = nothing
    if guidance_scale > 0
        negatives = negative_prompt === nothing ? fill("", length(prompts)) :
                    negative_prompt isa AbstractString ? [negative_prompt] :
                    collect(negative_prompt)
        length(negatives) == length(prompts) ||
            error("got $(length(prompts)) prompts and $(length(negatives)) negative prompts")
        negative = encode(negatives)
    end
    return p(encode(prompts); negative_prompt_embeds = negative, guidance_scale, kwargs...)
end

function (p::ZImagePipeline)(prompt_embeds::AbstractVector{<:AbstractMatrix};
        negative_prompt_embeds = nothing, height::Integer = 1024, width::Integer = 1024,
        num_inference_steps::Integer = 50, sigmas = nothing, guidance_scale = 5.0,
        cfg_normalization::Real = false, cfg_truncation = 1.0, latents = nothing,
        rng::AbstractRNG = Random.default_rng(), output_type::Symbol = :image)
    output_type in (:image, :latent) ||
        error("`output_type` must be :image or :latent, not $(repr(output_type))")
    f = 2 * vae_scale_factor(p)
    height % f == 0 && width % f == 0 ||
        error("`height` and `width` must be divisible by $f, not $height and $width")
    b = length(prompt_embeds)
    guided = guidance_scale > 0
    if guided
        negative_prompt_embeds === nothing &&
            error("classifier-free guidance (`guidance_scale > 0`) needs `negative_prompt_embeds`")
        length(negative_prompt_embeds) == b ||
            error("got $b prompt embeddings and $(length(negative_prompt_embeds)) negative ones")
    end
    T = eltype(first(prompt_embeds))
    L = promote_type(Float32, T)
    shape = (2 * (width ÷ f), 2 * (height ÷ f), p.transformer.in_channels, b)
    latents = latents === nothing ? randn(rng, Float32, shape) : latents
    size(latents) == shape ||
        error("`latents` must have size $shape for these `height`, `width` and prompts")
    x = copyto!(similar(first(prompt_embeds), L, shape), latents)
    p.scheduler.use_dynamic_shifting &&
        error("the Metalhead port of ZImagePipeline doesn't support schedulers with `use_dynamic_shifting = true`")
    sigmas = something(sigmas, z_image_sigmas(num_inference_steps))
    timesteps, sigmas = set_timesteps(p.scheduler, num_inference_steps; sigmas)
    for (i, t) in enumerate(timesteps)
        tnorm = (1000.0f0 - t) / 1000.0f0
        timestep = copyto!(similar(x, Float32, b), fill(tnorm, b))
        truncated = cfg_truncation !== nothing && cfg_truncation <= 1 &&
                    tnorm > cfg_truncation
        scale = truncated ? 0.0 : guidance_scale
        xt = convert_like(first(prompt_embeds), x)
        v = if guided && scale > 0
            out = p.transformer(cat(xt, xt; dims = 4), [timestep; timestep],
                [prompt_embeds; negative_prompt_embeds])
            pos, neg = L.(out[:, :, :, 1:b]), L.(out[:, :, :, (b + 1):end])
            guided_flow(pos, neg, L(scale), L(cfg_normalization))
        else
            L.(p.transformer(xt, timestep, prompt_embeds))
        end
        x = step(p.scheduler, -v, sigmas[i], sigmas[i + 1], x)
    end
    output_type === :latent && return x
    vae = p.vae
    z = convert_like(first(prompt_embeds), x) ./ T(vae.scaling_factor) .+
        T(something(vae.shift_factor, 0))
    return clamp.(decode(vae, z) ./ 2 .+ T(0.5), 0, 1)
end

# diffusers' default noise levels, `torch.linspace(1, 1 / n, n)` in float32, which torch
# computes with fused multiply-adds from both ends.
function z_image_sigmas(n::Integer)
    start, stop = 1.0f0, Float32(1 / n)
    step = (stop - start) / Float32(max(n - 1, 1))
    return [i < n ÷ 2 ? fma(step, Float32(i), start) : fma(-step, Float32(n - 1 - i), stop)
            for i in 0:(n - 1)]
end

# `pos + scale * (pos - neg)`, with the norm of each image at most `normalization` times
# that of `pos` when `normalization > 0`.
function guided_flow(pos, neg, scale, normalization)
    pred = pos .+ scale .* (pos .- neg)
    normalization > 0 || return pred
    norms(y) = sqrt.(sum(abs2, y; dims = 1:(ndims(y) - 1)))
    limit = norms(pos) .* normalization
    actual = norms(pred)
    return pred .* ifelse.(actual .> limit, limit ./ actual, one(eltype(pred)))
end
