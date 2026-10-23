# AGENTS.md

Metalhead.jl provides Flux ports of Hugging Face Hub models: the vision models of
[timm](https://github.com/huggingface/pytorch-image-models), language models of
[transformers](https://github.com/huggingface/transformers) (Llama and Qwen for now), and
text-to-image pipelines of [diffusers](https://github.com/huggingface/diffusers) (FLUX.2
[klein] for now). Each port mirrors the PyTorch module tree, loads the original safetensors
weights by name, and is checked numerically against PyTorch. `create_model` is the entry point.

## Layout

- `src/hub/`: Hub downloads, safetensors checkpoints, the strict weight loader
  (`loadweights.jl`), the architecture registry (`ArchSpec`, `ConfigFormat`, `register!`) and
  the public API (`api.jl`). Nothing there is specific to timm or to vision.
- `src/Timm/`: one file per timm family. Ports of `timm.layers` classes go in `layers.jl`.
  `config.jl` is timm's `ConfigFormat`, the template for any other library: it says how a
  `config.json` names an architecture and its arguments. A library gets its own `src/<Name>/`
  module, and its architecture names must not clash with those already registered.
- `src/Transformers/`: one file per transformers model, and `config.jl`, its `ConfigFormat`.
  Blocks that models share (RMSNorm, the rotary tables, the tied output head) go in `layers.jl`.
- `src/Diffusers/`: diffusers models (`transformer_flux2.jl`, `autoencoder_kl_flux2.jl`, the
  autoencoders' encoder and decoder in `vae.jl`), schedulers and pipelines. `config.jl` has two
  `ConfigFormat`s: a model's `config.json` and a pipeline's `model_index.json`, whose components
  load from subfolders of the repository. The tokenizer of a pipeline loads through the
  HuggingFaceTokenizers.jl extension (`ext/`).
- `src/legacy/`: the hand-written models of Metalhead 0.9 (`ResNet(50)`, `Metalhead.Layers`,
  `pretrain = true`). Deprecated, untested, and removed in 0.11: don't extend them.
- `test/`: run with ParallelTestRunner. Each file is a `@testset` in its own module. Shared
  helpers are in `test/parity.jl` and `test/parity_setup.jl`; tiny PyTorch fixtures in
  `test/fixtures/`, real-weight references and parameter counts in `test/references/`.
- `porting/`: the Python (uv) project that writes the fixtures and references. The package
  and its tests never run Python.
- `skills/port-hf-model/SKILL.md`: the porting workflow, conventions and numerical pitfalls.
- `docs/`: Documenter. `docs/make.jl` generates `index.md` (from the README) and `models.md`
  (from the registry), and checks that the migration guide names only ported architectures.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.test()'                                  # all tests
julia --project -e 'using Pkg; Pkg.test(test_args = ["models/resnet"])'     # by path prefix
julia --project -e 'using Pkg; Pkg.test(test_args = ["--list"])'            # list test files
METALHEAD_TEST_NETWORK=true julia --project -e 'using Pkg; Pkg.test()'     # also download weights
METALHEAD_TEST_NETWORK=true METALHEAD_TEST_LARGE=true julia --project -e 'using Pkg; Pkg.test(test_args = ["models/flux2"])'  # 16 GB of FLUX.2
```

`Pkg.test` builds a fresh environment each time. For quick iteration, load Metalhead once in a
persistent Julia session and `include` `test/parity.jl` (`includet` with Revise; see the skill).

## Conventions

- To add, fix or debug a ported model, follow `skills/port-hf-model/SKILL.md`. A port is done
  when its fixtures, parameter counts and references pass.
- Weight keys come from struct field names: keep them equal to the torch attribute names.
- Code style is in `.JuliaFormatter.toml` (SciML style, 92 columns).
- Record user-visible changes under `[Unreleased]` in `CHANGELOG.md`.
