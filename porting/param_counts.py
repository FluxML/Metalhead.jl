"""Record the parameter count and state_dict size of timm architectures.

    uv run param_counts.py <family> <architecture> ...

Writes test/references/param_counts/<family>.json, which the Julia tests use to check the
architecture tables of the ports without downloading weights.
"""
import json
import pathlib
import sys

import timm

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    family, *archs = sys.argv[1:]
    out = ROOT / "test" / "references" / "param_counts" / f"{family}.json"
    counts = json.loads(out.read_text()) if out.exists() else {}
    for arch in archs:
        model = timm.create_model(arch, pretrained=False)
        keys = [k for k in model.state_dict() if not k.endswith("num_batches_tracked")]
        counts[arch] = {"num_params": sum(p.numel() for p in model.parameters()), "num_tensors": len(keys)}
        print(arch, counts[arch], "num_classes =", model.num_classes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(sorted(counts.items())), indent=1) + "\n")


if __name__ == "__main__":
    main()
