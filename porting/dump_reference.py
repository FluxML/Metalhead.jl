"""Write reference outputs of a real pretrained checkpoint.

    uv run dump_reference.py timm/<model> [<org>/<model> ...] [--revision SHA]

Writes test/references/<org>--<model>.safetensors for each repo, with the logits and
pre-logits features of the checkpoint on `reference_input`, computed in float32. The metadata records the repo and
the commit sha, which the Julia test pins.

For a transformers causal LM, the input is `PROMPTS` tokenized by the repo's tokenizer and cut to
a common length (`input_ids`, 0-based), and the outputs are the hidden states after the final
norm, the logits of the last position, and the greedy continuation of the first prompt
(`generated_ids`), in float32, so that the Julia tests need neither Python nor a tokenizer.
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


PROMPTS = ["The capital of France is Paris, and the capital of Italy is",
           "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]


def dump_reference(repo, revision, batch):
    sha = HfApi().model_info(repo, revision=revision).sha
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    if "model_type" in config:
        return dump_lm_reference(repo, sha)
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


def dump_lm_reference(repo, sha, max_new_tokens=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(repo, revision=sha, dtype=torch.float32,
                                                 attn_implementation="sdpa").eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha)
    ids = [tokenizer(p)["input_ids"] for p in PROMPTS]
    n = min(map(len, ids))
    input_ids = torch.tensor([i[:n] for i in ids])
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
        logits = model.lm_head(hidden[:, -1, :])
        generated = model.generate(input_ids[:1], max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)

    out = ROOT / "test" / "references" / (repo.replace("/", "--") + ".safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"repo": repo, "revision": sha, "prompts": json.dumps(PROMPTS), **versions()}
    tensors = {"input_ids": input_ids, "hidden_states": hidden.contiguous(),
               "last_logits": logits.contiguous(), "generated_ids": generated.contiguous()}
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB, greedy continuation "
          f"{tokenizer.decode(generated[0, n:])!r}")


if __name__ == "__main__":
    main()
