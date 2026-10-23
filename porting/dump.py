"""Write a tiny parity fixture for a Metalhead port.

    uv run dump.py specs/<name>.json [--out DIR] [--all-taps]

Optional spec keys: "shrink": k divides every channel count by k (see `shrunk_channels` in
common.py; the Julia side applies `shrink_channels` from config.json), and
"weight_dtype": "float16" stores the weights in half precision, and "tap_depth": d records the
modules with at most d dots in their name (default 1), to keep large fixtures small.

The fixture directory looks like a Hub repository (config.json + model.safetensors), so the Julia
side loads it with `create_model(dir; pretrained = true)`, plus acts.safetensors with the input,
logits, features and module outputs ("taps") of the PyTorch model, computed in float64.
"""
import argparse
import json
import pathlib

import torch
from safetensors.torch import save_file

from common import build, classifier_name, rerandomize, state_dict_f32, versions, write_json

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec")
    p.add_argument("--out", help="output directory (default: test/fixtures/<spec name>)")
    p.add_argument("--all-taps", action="store_true", help="record every module, not only depth <= 2")
    p.add_argument("--max-bytes", type=int, default=1_000_000)
    p.add_argument("--tap-dtype", choices=["float16", "float32"],
                   help="storage type of the taps (default: float16, or float32 with --all-taps)")
    args = p.parse_args()

    spec_path = pathlib.Path(args.spec)
    spec = json.loads(spec_path.read_text())
    out = pathlib.Path(args.out) if args.out else ROOT / "test" / "fixtures" / spec_path.stem
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(spec.get("seed", 0))
    model = build(spec).eval()
    rerandomize(model, spec.get("seed", 0))
    weights = state_dict_f32(model)
    if spec.get("weight_dtype") == "float16":  # halves the file; torch gets the rounded values
        weights = {k: v.half() if v.is_floating_point() else v for k, v in weights.items()}
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in weights.items()})
    model = model.double()

    tap_dtype = getattr(torch, args.tap_dtype or ("float32" if args.all_taps else "float16"))
    taps, order, counts = {}, [], {}

    def hook(name):
        def fn(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            counts[name] = counts.get(name, 0) + 1
            for i, o in enumerate(outs):
                if isinstance(o, torch.Tensor):
                    key = f"{name}#{counts[name]}" + (f".{i + 1}" if len(outs) > 1 else "")
                    taps["tap/" + key] = o.detach().to(tap_dtype).contiguous()
                    order.append(key)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if n and (args.all_taps or n.count(".") <= spec.get("tap_depth", 1))]
    gen = torch.Generator().manual_seed(spec.get("seed", 0) + 1)
    x = torch.randn(spec["input_shape"], generator=gen, dtype=torch.float64)
    with torch.no_grad():
        logits = model(x)
        for h in handles:
            h.remove()
        features = model.forward_features(x)

    kwargs = dict(spec.get("kwargs", {}))
    num_classes = kwargs.pop("num_classes", getattr(model, "num_classes", None))
    config = {
        "architecture": spec["architecture"],
        "num_classes": num_classes,
        "model_args": kwargs,
        "pretrained_cfg": {"input_size": spec["input_shape"][1:], "classifier": classifier_name(model)},
    }
    if spec.get("shrink"):
        config["shrink"] = spec["shrink"]
    write_json(out / "config.json", config)
    save_file(weights, out / "model.safetensors")
    acts = {"input": x, "output": logits.contiguous(), "features": features.contiguous(), **taps}
    meta = {"spec": json.dumps(spec), "call_order": json.dumps(order),
            "num_params": str(sum(p.numel() for p in model.parameters())), **versions()}
    save_file(acts, out / "acts.safetensors", metadata=meta)

    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: {len(weights)} tensors, {len(taps)} taps, {size / 1e6:.2f} MB")
    if size > args.max_bytes and not args.out:
        raise SystemExit(f"fixture is larger than {args.max_bytes} bytes; shrink the spec")


if __name__ == "__main__":
    main()
