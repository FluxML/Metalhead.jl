# AGENTS.md

Metalhead.jl provides Flux ports of Hugging Face Hub models, for now the vision models of
[timm](https://github.com/huggingface/pytorch-image-models). Each port mirrors the PyTorch
module tree, loads the original safetensors weights by name, and is checked numerically
against PyTorch. `create_model` is the entry point.

## Layout

- `src/hub/`: Hub downloads, safetensors checkpoints, the strict weight loader
  (`loadweights.jl`), the architecture registry (`ArchSpec`, `register!`) and the public API
  (`api.jl`).
- `src/Timm/`: one file per timm family. Ports of `timm.layers` classes go in `layers.jl`.
- `src/legacy/`: the hand-written models of Metalhead 0.9 (`ResNet(50)`, `Metalhead.Layers`,
  `pretrain = true`). Deprecated, untested, and removed in 0.11: don't extend them.
- `test/`: run with ParallelTestRunner. Each file is a `@testset` in its own module. Shared
  helpers are in `test/parity.jl` and `test/parity_setup.jl`; tiny PyTorch fixtures in
  `test/fixtures/`, real-weight references and parameter counts in `test/references/`.
- `porting/`: the Python (uv) project that writes the fixtures and references. The package
  and its tests never run Python.
- `skills/port-hf-model/SKILL.md`: the porting workflow, conventions and numerical pitfalls.
  `.claude/skills` links to `skills/`.
- `docs/`: Documenter. `docs/make.jl` generates `index.md` (from the README) and `models.md`
  (from the registry), and checks that the migration guide names only ported architectures.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.test()'                                  # all tests
julia --project -e 'using Pkg; Pkg.test(test_args = ["models/resnet"])'     # by path prefix
julia --project -e 'using Pkg; Pkg.test(test_args = ["--list"])'            # list test files
METALHEAD_TEST_NETWORK=true julia --project -e 'using Pkg; Pkg.test()'     # also download weights
```

`Pkg.test` builds a fresh environment each time. For quick iteration, load Metalhead once in a
persistent Julia session and `includet` `test/parity.jl` (see the skill).

## Conventions

- To add, fix or debug a ported model, follow `skills/port-hf-model/SKILL.md`. A port is done
  when its fixtures, parameter counts and references pass.
- Weight keys come from struct field names: keep them equal to the torch attribute names.
- Code style is in `.JuliaFormatter.toml` (SciML style, 92 columns).
- Record user-visible changes under `[Unreleased]` in `CHANGELOG.md`.
