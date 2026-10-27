# Porting tools

Python side of the Metalhead ports: it produces the fixtures the Julia tests compare against.
This folder is not part of the Julia package API, and the Julia tests never run Python.

```sh
cd porting
uv sync           # torch (CPU), timm, transformers, diffusers, sentence-transformers, safetensors
uv run inspect_model.py resnet18          # module tree and state_dict of a timm model
uv run dump.py specs/resnet_basic.json specs/resnext.json     # tiny fixtures -> test/fixtures/<name>/
julia --project=.. check.jl specs/resnet_basic.json specs/resnext.json   # parity reports, one session
uv run dump.py specs/resnet_basic.json --all-taps --out out/resnet_basic   # debug: every module, float32
julia --project=.. check.jl out/resnet_basic
uv run param_counts.py resnet resnet18 resnet50   # -> test/references/param_counts/resnet.json
uv run dump_reference.py timm/resnet18.a1_in1k   # real weights -> test/references/
uv run dump_timm_transform.py                    # timm's eval transform (EvalTransform), Pillow's resize
uv run dump.py specs/llama_tiny.json             # a transformers spec ("library": "transformers")
uv run dump_reference.py HuggingFaceTB/SmolLM2-135M   # a language model: tokenized prompts
uv run dump.py specs/qwen3_5_vision_tiny.json    # a vision encoder: random patches of "grid_thw"
uv run dump_processor.py                         # processors (SAM 3, Qwen3.5, SigLIP, CLIP), through AutoProcessor
uv run dump.py specs/siglip2_tiny.json           # a dual encoder ("inputs": texts, images or patches)
uv run dump_reference.py google/siglip2-base-patch16-naflex   # an image and texts, through the processor
uv run dump.py specs/qwen3_5_tiny.json           # a vision-language model ("images")
uv run dump_reference.py Qwen/Qwen3.5-0.8B       # with a vision_config: also an image prompt
uv run dump_reference.py Qwen/Qwen3-30B-A3B --layers 4 --selective-shards   # first 4 layers only

uv run dump_reference.py Qwen/Qwen3.8-27B --layers 8   # too large: its first 8 layers
uv run dump.py specs/transformer_flux2_tiny.json specs/flux2_klein_tiny.json   # diffusers model, pipeline
uv run dump_reference.py black-forest-labs/FLUX.2-klein-4B/vae   # a component in a subfolder
uv run dump_reference.py black-forest-labs/FLUX.2-klein-4B       # a pipeline, end to end
uv run dump.py specs/st_qwen3_lasttoken.json     # a sentence-transformers model, saved by it
uv run dump_reference.py Qwen/Qwen3-Embedding-0.6B   # with a modules.json: its `encode`
julia --project=.. check_reference.jl ../test/references/timm--resnet18.a1_in1k.safetensors
julia --project=.. -e 'using Pkg; Pkg.test(test_args = ["models/resnet"])'   # one family's tests
```

Optional keys of a spec, for models too large for a fixture of 1 MB:

- `"shrink": k` builds the timm model with every channel count divided by `k` (the image channels
  and `num_classes` stay). `dump.py` records it in `config.json`, and `check_fixture` rebuilds the
  full-size Julia model, shrinks it with `shrink_channels` (test/parity.jl) and loads the weights.
- `"weight_dtype": "float16"` stores the weights in half precision (PyTorch uses the rounded values).
- `"tap_depth": d` records only the modules with at most `d` dots in their name (default 1;
  `-1` records none).

Feature intermediates are always stored in float64, with timm's tap metadata, independently
of `tap_depth`. Models without `forward_intermediates` use hooks at their `feature_info`
modules. The Julia harness checks these tensors at 1e-8 relative error and compares the tap
table, selected indices and early stopping.

The workflow for porting a model is in `skills/port-hf-model/SKILL.md`.
