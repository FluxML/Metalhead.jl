"""Write reference outputs of a real pretrained checkpoint.

    uv run dump_reference.py timm/<model> [timm/<model> ...] [--revision SHA]

Writes test/references/<org>--<model>.safetensors for each repo, with the logits and
pre-logits features of the checkpoint on `reference_input`, computed in float32. The metadata records the repo and
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
    p.add_argument("repos", nargs="+", metavar="repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()
    for repo in args.repos:
        dump_reference(repo, args.revision, args.batch)


def dump_reference(repo, revision, batch):
    sha = HfApi().model_info(repo, revision=revision).sha
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    weights = hf_hub_download(repo, "model.safetensors", revision=sha)
    # the repository's pretrained cfg, as timm uses when loading from the Hub: with the
    # architecture's default cfg, a checkpoint of another tag could lose its classifier
    cfg = dict(config["pretrained_cfg"], file=weights, custom_load=False,
               num_classes=config["num_classes"])
    model = timm.create_model(config["architecture"], pretrained=True, num_classes=config["num_classes"],
                              pretrained_cfg_overlay=cfg, **config.get("model_args", {}))
    model.eval()

    shape = [batch, *config["pretrained_cfg"]["input_size"]]
    x = reference_input(shape)
    with torch.no_grad():
        features = model.forward_features(x)
        pre_logits = model.forward_head(features, pre_logits=True)
        logits = model.forward_head(features)

    out = ROOT / "test" / "references" / (repo.replace("/", "--") + ".safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"repo": repo, "revision": sha, "input_shape": json.dumps(shape), **versions()}
    save_file({"logits": logits.contiguous(), "pre_logits": pre_logits.contiguous()}, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
