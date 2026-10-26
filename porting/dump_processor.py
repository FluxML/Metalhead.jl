"""Write the fixtures and references of processors, through the full upstream path.

    uv run dump_processor.py [sam3]    # all by default

sam3: the CLIP tokenizer of SAM 3 as `AutoTokenizer.from_pretrained(repo)` loads it, against
which test/processors/tokenizers.jl checks HuggingFaceTokenizers' tokenizer of the repository
and the padding of `Metalhead.pad_ids`. Writes

- test/fixtures/processors/unsloth--sam3.1/: the small JSON files of its tokenizer (not
  tokenizer.json);
- test/references/processors/sam3.safetensors: "tokenizer/..." cases, which call the
  tokenizer (padding, truncation, decoding).
"""
import json
import pathlib
import shutil
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

from common import versions

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    names = sys.argv[1:] or ["sam3"]
    for name in names:
        {"sam3": sam3}[name]()


def copy_config_files(repo, revision, files):
    """The small JSON files of a repository, in test/fixtures/processors/<org>--<repo>/."""
    out = ROOT / "test" / "fixtures" / "processors" / repo.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copyfile(hf_hub_download(repo, file, revision=revision), out / file)
    return out


SAM3_REPO, SAM3_REVISION = "unsloth/sam3.1", "cd3538905f98152ba184d2791fc5aa5d06e2ae45"
SAM3_FILES = ["tokenizer_config.json", "special_tokens_map.json"]


def sam3():
    copy_config_files(SAM3_REPO, SAM3_REVISION, SAM3_FILES)
    tokenizer = AutoTokenizer.from_pretrained(SAM3_REPO, revision=SAM3_REVISION)
    tensors = {}
    meta = {"repo": SAM3_REPO, "revision": SAM3_REVISION, **versions()}
    long = "a photo of " + " and ".join(["a small red cat"] * 8)
    sam3_tokenizer(tokenizer, tensors, meta, long)

    out = ROOT / "test" / "references" / "processors" / "sam3.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {out.stat().st_size / 1e3:.1f} kB")


def sam3_tokenizer(tokenizer, tensors, meta, long):
    """The CLIP tokenizer of SAM 3 alone: each prompt's ids with and without special tokens,
    padding, truncation (which keeps the special tokens) and decoding, as
    "tokenizer/<name>/..."."""
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(hf_hub_download(SAM3_REPO, "tokenizer.json", revision=SAM3_REVISION))
    texts = ["a cat", "Two DOGS, on a sofa!", long]
    meta["tokenizer/texts"] = json.dumps(texts)
    for i, text in enumerate(texts):
        tensors[f"tokenizer/ids/{i + 1}"] = torch.tensor(tokenizer(text)["input_ids"])
        tensors[f"tokenizer/content_ids/{i + 1}"] = torch.tensor(
            tokenizer(text, add_special_tokens=False)["input_ids"])
    calls = {
        "longest": dict(text=texts[:2], padding=True),
        "max_length": dict(text=texts[:2], padding="max_length", max_length=12),
        "left": dict(text=texts[:2], padding="max_length", max_length=12, padding_side="left"),
        "truncate": dict(text=texts, padding=True, truncation=True, max_length=10),
        "truncate_only": dict(text=[texts[2]], max_length=10),  # max_length alone truncates
        "model_max_length": dict(text=texts, padding=True, truncation=True),
        "no_special": dict(text=texts[:2], padding=True, add_special_tokens=False),
    }
    for name, kwargs in calls.items():
        out = tokenizer(**kwargs, return_tensors="pt")
        tensors[f"tokenizer/{name}/input_ids"] = out["input_ids"].contiguous()
        tensors[f"tokenizer/{name}/attention_mask"] = out["attention_mask"].contiguous()
        meta[f"tokenizer/{name}"] = json.dumps(dict(kwargs, text=[texts.index(t) + 1
                                                                  for t in kwargs["text"]]))
    ids = tokenizer(texts[1])["input_ids"]
    decoded = {f"{skip}": dict(text=tokenizer.decode(ids, skip_special_tokens=skip),
                               raw=raw.decode(ids, skip_special_tokens=skip))
               for skip in (False, True)}
    meta["tokenizer/decode"] = json.dumps(dict(ids=2, **decoded))


if __name__ == "__main__":
    main()
