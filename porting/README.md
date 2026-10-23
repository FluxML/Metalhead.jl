# Porting tools

Python side of the Metalhead ports: it produces the fixtures the Julia tests compare against.
This folder is not part of the Julia package API, and the Julia tests never run Python.

```sh
cd porting
uv sync                                   # torch (CPU), timm, safetensors
uv run inspect_model.py resnet18          # module tree and state_dict of a timm model
uv run dump.py specs/resnet_basic.json    # tiny fixture -> test/fixtures/resnet_basic/
julia --project=.. check.jl specs/resnet_basic.json      # per-module parity report
uv run dump.py specs/resnet_basic.json --all-taps --out out/resnet_basic   # debug: every module, float32
julia --project=.. check.jl out/resnet_basic
uv run param_counts.py resnet resnet18 resnet50   # -> test/references/param_counts/resnet.json
uv run dump_reference.py timm/resnet18.a1_in1k   # real weights -> test/references/
julia --project=.. check_reference.jl ../test/references/timm--resnet18.a1_in1k.safetensors
```

The workflow for porting a model is in `.claude/skills/port-hf-model/SKILL.md`.
