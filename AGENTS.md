# AGENTS.md

Metalhead.jl provides Flux ports of Hugging Face Hub models: the vision models of
[timm](https://github.com/huggingface/pytorch-image-models), language models of
[transformers](https://github.com/huggingface/transformers) (Llama, Qwen, Gemma 3 and
Granite for now), text-to-image pipelines of
[diffusers](https://github.com/huggingface/diffusers) (FLUX.2 [klein] and Z-Image for now),
and the text embeddings of
[sentence-transformers](https://github.com/huggingface/sentence-transformers) (its modules
over the transformers models: Qwen3-Embedding for now). Each port mirrors the PyTorch module
tree, loads the original safetensors weights by name, and is checked numerically against
PyTorch, and so is the processor of each checkpoint, its inference pre- and post-processing.
`create_model` and `create_processor` are the entry points.

## Scope

Port what a checkpoint determines and validate it against PyTorch; don't reimplement the
general-purpose machinery of timm, transformers, diffusers and sentence-transformers. A
feature belongs when the checkpoint determines it (its configuration, its weights, or
upstream code specific to the architecture), and getting it wrong would silently change the
results.

- In: architectures and weights, the feature interface, the processors (the inference pre-
  and post-processing a checkpoint sets), the schedulers and sampling loops of ported
  pipelines, the pooling, projections, normalization and prompts of sentence-transformers
  models, the key-value cache.
- Out: tokenization algorithms (HuggingFaceTokenizers.jl, through an extension), chat
  templates, decoding strategies, training utilities, augmentation, data loading, task APIs,
  quantization, LoRA, sentence-transformers' evaluation and similarity utilities.

General-purpose code that parity needs stays internal. `docs/src/scope.md` has the
reasons.

## Layout

- `src/hub/`: Hub downloads, safetensors checkpoints, the strict weight loader
  (`loadweights.jl`), the architecture registry (`ArchSpec`, `AbstractConfigFormat`,
  `register!`), the public API (`api.jl`), and `processors.jl`: `create_processor`, the
  interface of processor classes (`@processor`, `register_processor!`,
  `standardize_settings`), AutoProcessor's lookup, the strict reading of settings and the
  generic post-processing functions; `tokenizers.jl`, the internal interface through which
  processors and pipelines use a tokenizer (`load_tokenizer`, `encode_ids`, `token_id`,
  `pad_token_id`, `pad_ids`), and `TextProcessor`, the processor of a repository whose
  processor is its tokenizer (a language model's). `images.jl` has the image layout that
  every processor takes (`image_array`, and its inverse `image_matrix`). Nothing there is
  specific to one library.
- `src/Timm/`: one file per timm family. Ports of `timm.layers` classes go in `layers.jl`.
  `config.jl` is timm's `AbstractConfigFormat`, the template for any other library: it says
  how a `config.json` names an architecture and its arguments. A library gets its own
  `src/<Name>/` module, and its architecture names must not clash with those already
  registered. `transforms.jl` is timm's eval transform (`EvalTransform`, the processor of
  every timm checkpoint), and `pil.jl` a port of Pillow's 8-bit resize.
- `src/Transformers/`: one file per transformers model, and `config.jl`, its
  `AbstractConfigFormat`. Blocks that models share (RMSNorm, the rotary tables, the tied
  output head) go in `layers.jl`. Processors go next to their models, one file per upstream
  class under transformers' file names (`image_processing_sam3.jl`, `processing_sam3.jl`);
  `image_processing_backends.jl` has the operations of the torchvision backend that image
  processors share. Tokenizers aren't ported.
- `src/Diffusers/`: diffusers models (`transformer_flux2.jl`, `transformer_z_image.jl`,
  `autoencoder_kl_flux2.jl`, `autoencoder_kl.jl`, the autoencoders' encoder and decoder in
  `vae.jl`), schedulers and pipelines. `config.jl` has two `AbstractConfigFormat`s: a
  model's `config.json` and a pipeline's `model_index.json`, whose components load from
  subfolders of the repository (a transformers base model, such as Z-Image's `Qwen3Model`,
  as the decoder of its `...ForCausalLM` checkpoint). The tokenizer of a pipeline is
  HuggingFaceTokenizers.jl's `Tokenizer` of its `tokenizer/` subfolder, which `create_model`
  loads through the extension, or once the extension is loaded (`DeferredTokenizer`).
- `src/SentenceTransformers/`: sentence-transformers models, one file per upstream module
  (`transformer.jl`, `pooling.jl`, `dense.jl`, `normalize.jl`) and `model.jl`, the
  `SentenceTransformer` with `encode`, `encode_query`, `encode_document` and `similarity`.
  `config.jl` has the `AbstractConfigFormat` of a `modules.json`, which `source_config`
  reads before the `config.json` of the same repository, so that `create_model` builds the
  `SentenceTransformer`, each module from its subfolder; it reads the modules'
  configurations strictly (`read_settings`). The Transformer module holds the base model of
  a ported transformers architecture (`AUTO_MODELS`, transformers' `AutoModel`) and the
  tokenizer of the repository; a SentenceTransformer prepares its inputs itself and has no
  processor.
- `ext/`: package extensions. HuggingFaceTokenizers.jl's `Tokenizer` is the tokenizer of a
  checkpoint: its extension loads it from `tokenizer.json`, implements the interface of
  `src/hub/tokenizers.jl` and adds `encode(model::SentenceTransformer, texts)` to
  HuggingFaceTokenizers' `encode`; ImageCore lets processors take images of colorants and
  gives `image_matrix`; CUDA.
- `src/legacy/`: the hand-written models of Metalhead 0.9 (`ResNet(50)`, `Metalhead.Layers`,
  `pretrain = true`). Deprecated, untested, and removed in 0.11: don't extend them.
- `test/`: run with ParallelTestRunner. Each file is a `@testset` in its own module. Shared
  helpers are in `test/parity.jl` and `test/parity_setup.jl`; tiny PyTorch fixtures in
  `test/fixtures/`, real-weight references and parameter counts in `test/references/`.
  `test/gpu/` runs the fixtures on a CUDA GPU, only with `METALHEAD_TEST_GPU=true`
  (helpers in `test/gpu_setup.jl`). `test/processors/` checks the processors, with the
  configuration files of real repositories in `test/fixtures/processors/` and references in
  `test/references/` (helpers in `test/processors_setup.jl`).
- `porting/`: the Python (uv) project that writes the fixtures and references
  (`dump_processor.py` and `dump_timm_transform.py` those of processors, through the full
  upstream path; `dump_sentence_transformers.py` the tiny sentence-transformers models,
  saved as sentence-transformers saves them, with a tokenizer of numbered words). The
  package and its tests never run Python.
- `skills/port-hf-model/SKILL.md`: the porting workflow, conventions and numerical pitfalls.
- `docs/`: Documenter. `docs/make.jl` generates `index.md` (from the README) and `models.md`
  (from the registry), and checks that the migration guide names only ported architectures.
  `docs/figures/make_figures.jl` makes the figures of `docs/src/assets/` with the real models
  (FLUX.2 needs a GPU of 16 GB); the docs build doesn't run it.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.test()'                                  # all tests
julia --project -e 'using Pkg; Pkg.test(test_args = ["models/resnet"])'     # by path prefix
julia --project -e 'using Pkg; Pkg.test(test_args = ["--list"])'            # list test files
METALHEAD_TEST_NETWORK=true julia --project -e 'using Pkg; Pkg.test()'     # also download weights
METALHEAD_TEST_NETWORK=true METALHEAD_TEST_LARGE=true julia --project -e 'using Pkg; Pkg.test(test_args = ["models/flux2"])'  # 16 GB of FLUX.2
METALHEAD_TEST_GPU=true julia --project -e 'using Pkg; Pkg.test(test_args = ["gpu"])'  # the fixtures on a CUDA GPU
```

`Pkg.test` builds a fresh environment each time. For quick iteration, load Metalhead once in a
persistent Julia session and `include` `test/parity.jl` (`includet` with Revise; see the skill).

## Conventions

- To add, fix or debug a ported model, follow `skills/port-hf-model/SKILL.md`. A port is done
  when its fixtures, parameter counts and references pass.
- Weight keys come from struct field names: keep them equal to the torch attribute names.
- Code style is in `.JuliaFormatter.toml` (SciML style, 92 columns).
- Record user-visible changes under `[Unreleased]` in `CHANGELOG.md`.
