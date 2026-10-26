# [Text-to-image generation](@id text_to_image)

Metalhead ports text-to-image pipelines from
[Hugging Face Diffusers](https://github.com/huggingface/diffusers), starting with
[FLUX.2 \[klein\]](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) of Black Forest
Labs, a flow-matching model of 4 billion parameters under the Apache 2.0 license. A diffusers
repository holds a pipeline: its `model_index.json` names the components, each in a subfolder
of its own, and `create_model` builds them all.

```julia
using Metalhead, Flux, HuggingFaceTokenizers
using CUDA, cuDNN                   # optional: for a GPU

pipe = create_model("black-forest-labs/FLUX.2-klein-4B"; pretrained = true, eltype = BFloat16)
pipe = gpu(pipe)

image = pipe("A photo of a red fox sitting in fresh snow, looking at the camera";
    num_inference_steps = 4)
```

The pipeline downloads 16 GB of weights, and needs as much memory in `BFloat16` (twice that
in `Float32`). FLUX.2 [klein] 4B is distilled to generate an image in 4 steps. Its base
version, `black-forest-labs/FLUX.2-klein-base-4B`, takes about 50 steps with classifier-free
guidance (`guidance_scale = 4`, the default), and is the one to fine-tune.

![A red fox in the snow, by FLUX.2 [klein] 4B](../assets/flux2-fox.jpg)

`image` has size `(width, height, 3, batch)`, with values in [0, 1]. ImageCore.jl (or
Images.jl) makes an image of it, and FileIO saves it:

```julia
using ImageCore, FileIO

img = colorview(RGB, permutedims(Float32.(cpu(image[:, :, :, 1])), (3, 2, 1)))
FileIO.save("fox.png", img)  # HuggingFaceTokenizers exports a `save` too
```

## Options

The keyword arguments of the pipeline are those of diffusers' `Flux2KleinPipeline`:

- `height` and `width` (1024 by default), multiples of 16.
- `num_inference_steps`, and `guidance_scale` for the base checkpoints.
- `latents`: the initial noise, of size `(width ÷ 16, height ÷ 16, 128, batch)`. By default it is
  drawn from `rng` (`Random.default_rng()`), so that `rng = Xoshiro(0)` makes an image
  reproducible. diffusers draws its noise with PyTorch's generator, so the same seed gives
  different images in the two libraries; the same `latents` give the same image.
- `output_type = :latent` returns the denoised latents of the autoencoder instead of the image.

A vector of prompts makes a batch of images.

## The steps of the pipeline

The prompt is formatted with the chat template of the Qwen3 text encoder, tokenized by the
pipeline's tokenizer, and truncated and padded to 512 tokens. The hidden states of three layers of the
text encoder are its embeddings:

```julia
using Metalhead.Diffusers: encode_prompt

embeds = encode_prompt(pipe, ["A photo of a red fox"])  # 7680 × 512 × 1
```

The pipeline also takes the embeddings, `pipe(embeds; num_inference_steps = 4)`, which saves
encoding the same prompt again and needs no tokenizer. `encode_prompt` also takes the token
indices of prompts tokenized otherwise, with their attention mask.

The components are fields of the pipeline: `pipe.text_encoder` is a Qwen3 language model (see
[Language models](@ref language_models)), `pipe.transformer` predicts the flow of the latents
at each step, `pipe.vae` decodes the final latents into an image, and `pipe.scheduler` holds the
noise levels of the steps. `pipe.tokenizer` is the tokenizer of the text encoder,
HuggingFaceTokenizers.jl's `Tokenizer`, which `gpu` and `bf16` leave as it is; it loads with
the pipeline when HuggingFaceTokenizers is loaded first, and is `nothing` otherwise, when
the pipeline takes embeddings only. A pipeline has no processor of its own: it prepares its
inputs itself.
`create_processor(repo; subfolder = "tokenizer")` loads the tokenizer alone, and each other
component loads on its own with `subfolder` too:

```julia
using Metalhead.Diffusers: encode, decode, mode

vae = create_model("black-forest-labs/FLUX.2-klein-4B"; subfolder = "vae", pretrained = true)
x = rand(Float32, 256, 256, 3, 1) .* 2 .- 1    # an image with values in [-1, 1]
latents = mode(encode(vae, x))                 # 32 × 32 × 32 × 1
y = decode(vae, latents)                       # 256 × 256 × 3 × 1
```

Images are WHCN arrays, as for the vision models. See the docstrings of
[`Metalhead.Diffusers.Flux2KleinPipeline`](@ref) and its components in the
[API reference](@ref) for their inputs and outputs.

## Precision

The pipeline computes in the element type of its weights, `eltype`. In `BFloat16` (or
`Float16`) the normalization layers keep their parameters and statistics in `Float32` and
normalize in `Float32`, and, as in diffusers, the timestep embeddings and the Euler steps of the
scheduler are computed in `Float32`. On an RTX 5090, FLUX.2 [klein] 4B in `BFloat16` takes 15 GB
of GPU memory and about 2.5 seconds for an image of 1024 × 1024 pixels in 4 steps.

On the CPU, `Float32` is the practical choice: there, matrix products and convolutions in
`BFloat16` are slow and accumulate in 16 bits.
