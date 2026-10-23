"""Write reference outputs of a real pretrained checkpoint.

    uv run dump_reference.py timm/<model> [--revision SHA]

Writes test/references/<org>--<model>.safetensors with the logits and pre-logits features
of the checkpoint on `reference_input`, computed in float32. The metadata records the repo and
the commit sha, which the Julia test pins.
"""
import argparse
import json
import pathlib

import timm
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

from common import reference_input, versions

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()

    sha = HfApi().model_info(args.repo, revision=args.revision).sha
    config = json.loads(pathlib.Path(hf_hub_download(args.repo, "config.json", revision=sha)).read_text())
    weights = hf_hub_download(args.repo, "model.safetensors", revision=sha)
    model = timm.create_model(config["architecture"], pretrained=True, num_classes=config["num_classes"],
                              pretrained_cfg_overlay=dict(file=weights, custom_load=False), **config.get("model_args", {}))
    model.eval()

    shape = [args.batch, *config["pretrained_cfg"]["input_size"]]
    x = reference_input(shape)
    with torch.no_grad():
        features = model.forward_features(x)
        pre_logits = model.forward_head(features, pre_logits=True)
        logits = model.forward_head(features)

    out = ROOT / "test" / "references" / (args.repo.replace("/", "--") + ".safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"repo": args.repo, "revision": sha, "input_shape": json.dumps(shape), **versions()}
    save_file({"logits": logits.contiguous(), "pre_logits": pre_logits.contiguous()}, out, metadata=meta)
    print(f"{out}: {args.repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
