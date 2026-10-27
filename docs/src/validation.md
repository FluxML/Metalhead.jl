# [How models are validated](@id validation)

Every model in Metalhead is a port of a PyTorch model, checked against it numerically. This
page describes what is checked, and what that guarantees.

## What a port is

A port is a Julia file in `src/Timm/` (timm), `src/Transformers/` (transformers) or
`src/Diffusers/` (diffusers) that rebuilds the PyTorch module tree with Flux layers.
Each PyTorch submodule becomes a field with the same name, so a checkpoint's `state_dict`
keys follow from the structure of the Flux model. Weights load by name, with no hand-written
mapping. Arrays use Flux's layout, which is PyTorch's with the dimensions reversed (NCHW
becomes WHCN). Dense weights are transposed and convolution kernels are flipped on loading.

Loading is strict. It fails if a checkpoint tensor is unused, if a model parameter is missing,
or if any shape differs. The error names the closest unused key, to help track down a
misnamed field.

## Tiny fixtures

For each code path of an architecture (each block type, stem, or optional branch), the port
has a *fixture* in `test/fixtures/`. It is made from a small version of the model:
fewer blocks, narrow channels, ten classes, a small input. `porting/dump.py` builds it in
PyTorch and re-randomises every parameter and batch-norm statistic. This matters because
default initialisations hide bugs: a layer scale near zero, or a batch norm initialised to
zero at the end of each residual branch, makes whole branches contribute nothing.

`dump.py` runs the model in float64 and stores the weights, the input and the outputs. In CI,
the Flux model loads the weights and runs the same input. Its logits and features must match
PyTorch's to a **relative error of 1e-8**. A tolerance this tight catches mistakes that looser
checks miss, such as the tanh approximation of GELU in place of the exact one, or a norm
epsilon of `1e-5` instead of `1e-6`.

The fixture also stores *taps*: the outputs of the modules near the top of the tree, whose
names have at most one dot (`stem`, `stages.0`, `blocks.3`, ...), in float16. The Flux model records the outputs of its modules
with the same names, so a failing fixture shows the first module that differs from PyTorch.
For debugging, `dump.py --all-taps` records every module.

Feature intermediates are compared at the tolerance of their stored precision (1e-8 for
float64), along with timm's channels, reductions and module names. For families without a
native `forward_intermediates`, `dump.py` records hooks at the modules in `feature_info`.
The tests check selected indices, early stopping and gradients through the selected taps.

For a language model, the fixture is a small configuration of the transformers class (two or
three layers, a hidden size of a few dozen and a vocabulary of a few dozen tokens), saved with
`save_pretrained` like a Hub repository, and its input is random token ids. The taps go down to
the attention, MLP and norms of each layer. transformers computes its RMSNorms and rotary
embeddings in float32, even in a float64 model. `dump.py` computes them in float64, so that the
1e-8 comparison tests the architecture rather than that rounding. One fixture pads a batch on
the right and passes its `attention_mask`, and each records the hidden states of every layer.

For a diffusers model, the fixture is a small configuration of the class, saved with
`save_pretrained`, run on random inputs: images for an autoencoder; image tokens, text tokens,
noise levels and position ids for the FLUX.2 transformer; latent images, timesteps and
captions of different lengths for the Z-Image transformer, which pads them and masks the
padding. diffusers computes its timestep and rotary embeddings (and the RMSNorms of Z-Image)
in float32, which `dump.py` computes in float64 too. A pipeline fixture is a
whole pipeline of tiny components (text encoder, transformer, autoencoder and scheduler), saved
like a Hub repository with a `model_index.json`. It stores the prompt embeddings of random
token ids, which must match to 1e-8, and the latents and images that the pipeline makes from
fixed noise in a few steps, which match to 1e-10: diffusers keeps the noise levels in float32,
in a float64 pipeline too. The Z-Image pipeline denoises its latents in float32 whatever the
eltype of its models, so `dump.py` runs a copy of its code with float64 in place of float32.
One of its fixtures has classifier-free guidance with the normalization and truncation
options, and prompts whose lengths differ.

Fixtures are under 1 MB each. They run without Python and without network access.

## References

For one or two real checkpoints per family, `porting/dump_reference.py` stores PyTorch's
float32 outputs for a deterministic input in `test/references/`, with the repository and the
commit of the weights. The Flux model, loaded from the Hub at that commit, must match them to
**1e-4 in float32**. Correct ports land between 1e-7 and 1e-5. These tests download the
weights, so they only run with `METALHEAD_TEST_NETWORK=true`.

For a language model, the input is two prompts tokenized by the repository's tokenizer. The
reference stores the token ids, the hidden states after the final norm, the logits of the last
position and transformers' greedy continuation of the first prompt. The Flux model must match
the hidden states and logits to 1e-4, and reproduce the continuation exactly. The tests need
neither Python nor a tokenizer.

Metalhead ports no tokenizer: HuggingFaceTokenizers.jl runs Hugging Face's tokenizers library
on the repository's `tokenizer.json` (see [Language models](@ref language_models)). With
`METALHEAD_TEST_NETWORK=true`, the test suite adds it to its environment and checks that the
tokenizer that `create_processor` loads gives the token ids of these references, and of the
FLUX.2 [klein] pipeline below, as transformers' tokenizers gave them, with the padding token
of `tokenizer_config.json`.

For FLUX.2 [klein] 4B, the references are the latents and image of a few images through its
autoencoder, the output of its transformer for random inputs, and an image of 256 × 256 pixels
generated in 4 steps from a prompt and fixed noise, with the token ids and the prompt
embeddings. The tests check the three to 1e-3: the autoencoder and the transformer land near
1e-5, and the latents of the pipeline near 1e-3, as rounding errors grow over the denoising steps, so that PyTorch's own float32 latents
differ from its float64 ones by 6e-4. In float64, the port and PyTorch generate the same latents
to 1e-12. The transformer and the pipeline download 8 and 16 GB, and run only with
`METALHEAD_TEST_LARGE=true` as well.

Z-Image-Turbo has the same references: its autoencoder, its transformer on latents of 48
patches and captions of 10 and 40 tokens, and an image of 256 × 256 pixels generated with
its settings, 9 steps without guidance. They also store PyTorch's parameter counts. The
autoencoder lands at 3e-5, the transformer at 6e-6, and the latents and image of the
pipeline at 3e-6 and 2e-6. The transformer and the pipeline download 25 and 33 GB, and take
as much memory in float32 on the CPU (the pipeline 42 GB).

## Processors

A processor (see [`create_processor`](@ref)) is checked against the full upstream path, from
the raw input to the inputs of the model, with the configuration files of real repositories.
The references are in `test/references/`. The tests need no Python, and only the checks
marked below as such need network access.

For timm, `porting/dump_timm_transform.py` stores the output of timm's
`create_transform(**resolve_data_config({}, pretrained_cfg = cfg))` on PIL images, for 14
configurations: the six interpolations of Pillow, the `"center"` and `"squash"` crop modes,
non-square input sizes, a `crop_pct` above 1 (which pads), the ImageNet, Inception and CLIP
statistics, and `use_test_size`. Thirteen run on seven small images (one grey, one RGBA), and
the fourteenth, that of `timm/convnext_tiny.fb_in1k`, on an image of 500 × 375 pixels.
[`Timm.EvalTransform`](@ref Metalhead.Timm.EvalTransform) must give the same `Float32`
values, **bit for bit**, which takes a port of Pillow's 8-bit resampling: it matches
`Image.resize` exactly for each of its six filters. With network access, the settings that
`create_processor` resolves for each of the 56 timm checkpoints in the references, with and
without `use_test_size`, must be those that timm resolves, and the logits of
`timm/convnext_tiny.fb_in1k` for its transformed image must match PyTorch's to 1e-4.

For transformers, `porting/dump_processor.py` calls the processor that
`AutoProcessor.from_pretrained(repo)` loads, on small images, prompts and boxes: 18 cases
for SAM 3 (`unsloth/sam3.1` and `jetjodh/sam3`) and 19 for Qwen3.5 (`Qwen/Qwen3.5-0.8B`),
which cover batches, grey and RGBA images, upscaling and downscaling, box prompts,
`do_resize`, `do_rescale` and `do_normalize` off, overridden settings, and Qwen2-VL's legacy
`min_pixels` and `max_pixels`. Every output must be transformers', with the same keys. For
8-bit images, `pixel_values` are equal **bit for bit**: the resize ports the fixed-point
antialiased resize of `uint8` tensors that torchvision, transformers' default backend, runs
on x86 CPUs with AVX2. For images of reals, which take a floating-point path, they match to
a relative error of **1e-6**. Token ids, attention masks, boxes, labels and image grids are
equal. The offline tests stand in for the tokenizer with one that gives the ids of
transformers' tokenizer; with `METALHEAD_TEST_NETWORK=true`, they also tokenize the strings
with HuggingFaceTokenizers.jl and the repository's `tokenizer.json`. SAM 3's
post-processing is checked on random outputs of the model: the same scores and boxes, to
floating-point precision, and the same masks.

Configuration files are read strictly, an unknown key with a warning, and the tests load
those of the fixtures without one. Pixels are compared after decoding: a JPEG decoded by
ImageIO can differ slightly from Pillow's decoding of the same file.

## Parameter counts

The test suite builds every registered timm architecture with its default arguments and
compares its number of parameters and tensors with timm's, recorded by `porting/param_counts.py`. This
checks that the registered defaults reproduce what `timm.create_model` builds, including for
architectures that are too large for a fixture.

## GPU

With `METALHEAD_TEST_GPU=true`, the test suite adds CUDA.jl and cuDNN to its environment and
also runs every fixture on a CUDA GPU in Float32, with scalar indexing disallowed. The
outputs must match the references to 1e-2, as cuDNN convolutions run in TF32 by default:
the fixtures differ by up to 3e-3. A few fixtures also check gradients and BFloat16 on the
GPU. cuDNN's 3 × 3 mean pooling with padding is wrong on maps of 1 × 1 or 2 × 2 pixels, so
the Inception v3, v4 and Res2Net fixtures are compared with the CPU on larger images there.
CI runs these tests on a GPU.

## What this guarantees

- An architecture listed by [`list_models`](@ref) computes what its Python library (timm,
  transformers or diffusers) computes, in eval mode, for the code paths its fixtures cover.
  The registry only includes architectures whose options the port implements. Other options
  raise an error rather than being silently ignored.
- Any Hub checkpoint of a registered architecture loads with its exact structure, because
  loading is strict. The checkpoints listed in the [model list](@ref models) were also
  compared with PyTorch outputs.

Not checked:

- Training-mode behaviour. Dropout and stochastic depth are random, and batch normalisation
  updates its statistics like Flux does.
- The initialisation of randomly initialised models, which only roughly follows timm.
- Training augmentation, which processors don't do, and the decoding of image files.
