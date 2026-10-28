# [Porting a model](@id contributing)

Contributions are welcome: bug reports and feature requests as
[issues](https://github.com/FluxML/Metalhead.jl/issues/new/choose), fixes and new models as
[pull requests](https://github.com/FluxML/Metalhead.jl/pulls). See also the
[FluxML contributing guide](https://github.com/FluxML/Flux.jl/blob/master/CONTRIBUTING.md).

This page is about adding a model. Metalhead admits models whose weights are on the
[Hugging Face Hub](https://huggingface.co/models). For now these are
[timm](https://github.com/huggingface/pytorch-image-models) vision models,
[transformers](https://github.com/huggingface/transformers) language, vision-language and
segmentation models, [diffusers](https://github.com/huggingface/diffusers) models and
pipelines, and the text embeddings of
[sentence-transformers](https://github.com/huggingface/sentence-transformers). A port is a Flux reimplementation of the PyTorch code, validated as described in
[How models are validated](@ref validation). It includes what the checkpoint determines
besides the model, such as its preprocessing, but not general-purpose machinery: see
[Scope](@ref scope). The registry and `create_model` are not specific to timm: a library
other than timm gets its own module, `src/<Name>/`, and a
[`Metalhead.AbstractConfigFormat`](@ref) saying how its `config.json` names an architecture,
like `src/Timm/config.jl`.

## Setup

The porting tools in [`porting/`](https://github.com/FluxML/Metalhead.jl/tree/master/porting)
are a Python project managed with [uv](https://docs.astral.sh/uv/). It pins timm,
transformers, diffusers, sentence-transformers and a CPU build of PyTorch. Python is only
needed to port a model: the package and its tests don't use it.

```sh
cd porting
uv sync
```

## With Claude Code

The repository includes a [Claude Code](https://claude.com/claude-code) skill,
[`port-hf-model`](https://github.com/FluxML/Metalhead.jl/blob/master/skills/port-hf-model/SKILL.md).
It describes the workflow below, the conventions, and the numerical pitfalls met in earlier
ports. Start Claude Code in the repository and ask it to read the skill before porting, for
example:

```
Read skills/port-hf-model/SKILL.md and follow its workflow to port timm's RegNet family
to Metalhead.
```

Review the result like any pull request: the port is done when its tests pass, and the tests
compare against PyTorch rather than relying on the agent's claims.

## The workflow

The skill is also the reference for porting by hand. In short:

1. **Inspect** the model. `uv run inspect_model.py resnet18` prints the PyTorch module tree
   and the `state_dict`. Read the timm source of the family, including the constructors'
   branches: some modules exist only for non-default arguments.
2. **Write the port** in `src/Timm/<family>.jl`, mirroring the module tree (see below).
3. **Register** each supported architecture with `register!(ArchSpec(...))`, with the default
   arguments of its timm entry point and `family = "<timm module>"`. The family identifier
   groups models in `list_models(; byfamily = true)` and the documentation.
4. **Make tiny fixtures**: one `porting/specs/<name>.json` per code path, then
   `uv run dump.py specs/<name>.json` and `julia --project=.. check.jl specs/<name>.json`
   until it prints PASS.
5. **Record parameter counts** with `uv run param_counts.py <family> <architectures...>`.
6. **Make references** for one or two real checkpoints with
   `uv run dump_reference.py timm/<repo>`, check them with `check_reference.jl`, and add
   `repo => sha` (printed as `repo@sha`) to the architecture's `references`.
7. **Add tests** in `test/models/<family>.jl`, following the existing ones, and run
   `julia --project -e 'using Pkg; Pkg.test(test_args = ["models"])'`. On a machine with
   an NVIDIA GPU, `METALHEAD_TEST_GPU=true` and `test_args = ["gpu"]` run the fixtures on
   it (see [How models are validated](@ref validation)).

A transformers model follows the same steps, with three differences. The port goes in
`src/Transformers/<model>.jl`, and registers the transformers class (`"LlamaForCausalLM"`),
whose builder takes the fields of `config.json`. Its specs have `"library": "transformers"`, a
`model_type` and the `config` fields (see `porting/specs/llama_tiny.json`). It has no parameter
counts: its configurations are checkpoints, which load strictly.

A diffusers model goes in `src/Diffusers/`, and registers its class (`"AutoencoderKLFlux2"`),
whose builder takes the arguments of its constructor. Its specs have `"library": "diffusers"`,
the `config` and the sizes of the `inputs` (see `porting/specs/transformer_flux2_tiny.json`). A
pipeline registers its class too, built from components in subfolders, and its fixture runs a
pipeline of tiny components end to end (`porting/specs/flux2_klein_tiny.json`). The references
of a component are named `<org>/<repo>/<subfolder>`.

A sentence-transformers model is the modules of a `modules.json` over a transformers model:
a new transformers model for embeddings is added to `AUTO_MODELS`
(`src/SentenceTransformers/transformer.jl`), and a new module is a file with a loader in
`MODULE_LOADERS`. Its specs have `"library": "sentence-transformers"`
(`porting/specs/st_qwen3_dense.json`), and the references are
`uv run dump_reference.py <org>/<repo>` of a repository with a `modules.json`.

A checkpoint whose processor class isn't ported yet needs its processor too, ported with the
model: one file per upstream class, under transformers' file names
(`image_processing_<model>.jl`, `processing_<model>.jl`), registered with
`register_processor!`. `uv run dump_processor.py` stores the outputs of the processor that
transformers' `AutoProcessor` loads, on small images and prompts, which the tests in
`test/processors/` compare with (timm's transform: `dump_timm_transform.py`).

When a fixture fails, `uv run dump.py specs/<name>.json --all-taps --out out/<name>` records
the output of every module, and `check.jl out/<name>` shows the first one that differs.

## Conventions

- **Mirror the PyTorch module tree.** Struct fields have the names of the PyTorch submodule
  attributes. `nn.Sequential` becomes a `Chain` with the same entries, keeping parameter-free
  placeholders so that indices line up. Weights then load by name.
- **Use Flux layers and NNlib functions.** Helpers that timm shares between families
  (`timm.layers`) are ported once, in `src/Timm/layers.jl`.
- **Define structs with `@concrete`** (from ConcreteStructs.jl) and `Flux.@layer`.
- **Keep timm's keyword arguments.** Builders accept timm's names and defaults, and raise an
  error for options the port doesn't implement.
- **Follow timm's API.** Each model defines `forward_features` and `forward_head`.
- **Use Flux's layout.** It is PyTorch's with the dimensions reversed, so reshapes carry over
  with reversed sizes.

## Pull request checklist

- Every fixture passes at 1e-8, and the references pass at 1e-4.
- The parameter counts of every registered timm architecture match timm.
- A gradient test runs.
- The file header names the timm, transformers or diffusers module and version it was ported
  from.
- Fixtures are under 1 MB each.
