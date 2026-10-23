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
has a *fixture* in `test/fixtures/`. It is made from a small version of the timm model:
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
noise levels and position ids for the FLUX.2 transformer. diffusers computes its timestep and
rotary embeddings in float32, which `dump.py` computes in float64 too. A pipeline fixture is a
whole pipeline of tiny components (text encoder, transformer, autoencoder and scheduler), saved
like a Hub repository with a `model_index.json`. It stores the prompt embeddings of random
token ids, which must match to 1e-8, and the latents and images that the pipeline makes from
fixed noise in a few steps, which match to 1e-10: diffusers keeps the noise levels in float32,
in a float64 pipeline too.

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

For FLUX.2 [klein] 4B, the references are the latents and image of a few images through its
autoencoder, the output of its transformer for random inputs, and an image of 256 × 256 pixels
generated in 4 steps from a prompt and fixed noise, with the token ids and the prompt
embeddings. The autoencoder and the transformer match to 1e-5, and the latents of the pipeline
to 1e-3: rounding errors grow over the denoising steps, so that PyTorch's own float32 latents
differ from its float64 ones by 6e-4. In float64, the port and PyTorch generate the same latents
to 1e-12. The transformer and the pipeline download 8 and 16 GB, and run only with
`METALHEAD_TEST_LARGE=true` as well.

## Parameter counts

The test suite builds every registered timm architecture with its default arguments and
compares its number of parameters and tensors with timm's, recorded by `porting/param_counts.py`. This
checks that the registered defaults reproduce what `timm.create_model` builds, including for
architectures that are too large for a fixture.

## What this guarantees

- An architecture listed by [`list_models`](@ref) computes what timm computes, in eval mode,
  for the code paths its fixtures cover. The registry only includes architectures whose
  options the port implements. Other options raise an error rather than being silently
  ignored.
- Any Hub checkpoint of a registered architecture loads with its exact structure, because
  loading is strict. The checkpoints listed in the [model list](@ref models) were also
  compared with PyTorch outputs.

Not checked:

- Training-mode behaviour. Dropout and stochastic depth are random, and batch normalisation
  updates its statistics like Flux does.
- The initialisation of randomly initialised models, which only roughly follows timm.
- Image preprocessing. [`Metalhead.Timm.pretrained_cfg`](@ref) returns the settings a checkpoint
  expects, but applying them is up to you.
