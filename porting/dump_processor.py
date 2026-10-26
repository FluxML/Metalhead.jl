"""Write the fixtures and references of processors, through the full upstream path.

    uv run dump_processor.py [sam3] [tokenizers]    # all by default

sam3: the CLIP tokenizer of SAM 3 as `AutoTokenizer.from_pretrained(repo)` loads it, for
src/Transformers/tokenization_utils_tokenizers.jl. Writes

- test/fixtures/processors/unsloth--sam3.1/: the small JSON files of its tokenizer (not
  tokenizer.json), from which the Julia tests load it offline;
- test/references/processors/sam3.safetensors: "tokenizer/..." cases, which call the
  tokenizer (padding, truncation, decoding).

tokenizers: the tokenizers of the validated language models, of the FLUX.2 [klein] pipeline,
of Qwen3.5, of SAM 3 and of GPT-2, as `AutoTokenizer.from_pretrained(repo)` loads them, for
src/Transformers/tokenization_utils_tokenizers.jl (see `tokenizers`).
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
    names = sys.argv[1:] or ["sam3", "tokenizers"]
    for name in names:
        {"sam3": sam3, "tokenizers": tokenizers}[name]()


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


# (repo, revision, subfolder) of each tokenizer: those of the language models that Metalhead
# validates, of the FLUX.2 [klein] pipeline, of Qwen3.5 (a Qwen3_5Tokenizer) and of SAM 3 (a
# CLIPTokenizer, whose fixture files `sam3` writes)
TOKENIZER_REPOS = [
    ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", ""),
    ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da", ""),
    ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987", ""),
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", ""),
    ("unsloth/gemma-3-270m", "edcf1fe394ee1168e7c5538afd104cfb1f7947e6", ""),
    ("ibm-granite/granite-3.3-2b-base", "c15dc681f4b2aeb4e576cdaae5434edfc17ed14d", ""),
    ("ibm-granite/granite-3.1-1b-a400m-base", "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29", ""),
    ("ibm-research/moe-7b-1b-active-shared-experts", "6194d7218c3e792b939b535f5640f9c32be9b442",
     ""),
    ("ibm-granite/granite-4.0-350m-base", "a50b46cef21c8a86b15f0496cb794487a78a910b", ""),
    ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", ""),
    ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294", "tokenizer"),
    (SAM3_REPO, SAM3_REVISION, ""),
    # not a validated language model: the defaults of GPT2Tokenizer, which its files don't set
    ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", ""),
]
# and config.json, whose `model_type` can decide the class of the tokenizer
TOKENIZER_FILES = ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "config.json"]
# strings on which tokenizer classes can differ: case, accents, unicode normal forms, whitespace
# runs, digits, contractions, and the special tokens of each family inside the text
TOKENIZER_TEXTS = [
    "The capital of France is",
    "Hello",
    "In 2024, 1234567 people paid $3.50 each; 12 + 34 = 46.",
    "  leading spaces,\ttabs\tand\nnew\n\nlines  ",
    "Ünïcödé: naïve café, 日本語のテキスト, emoji 🦊🔥, Ελληνικά.",
    "def f(x):\n    return x**2  # comment\n",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "<bos>text with <eos> inline <pad> tokens <start_of_image><image_soft_token>",
    "<|begin_of_text|>Llama <|end_of_text|> and <|endoftext|> <fim_prefix>",
    "don't won't it's I'm we're they've",
    "HeLLo WORLD, MiXeD CaSe",
    "Ça va? Ñandú, Åsa, Zoë, İstanbul, STRASSE ß, ﬁ ligature, ＡＢＣ full width",
    "é combining vs é precomposed,  nbsp em​zero-width",
    "multiple     spaces\t\t\ttabs\n\n\nnewlines \r\n crlf   end   ",
    "",
    "<mask> token <mask>",
]


def tokenizers():
    """test/references/processors/tokenizers.safetensors: for each tokenizer of
    `TOKENIZER_REPOS`, under the key `<org>--<repo>[--<subfolder>]`, the ids of each text of
    `TOKENIZER_TEXTS` with and without special tokens ("<key>/ids/<i>", "<key>/content_ids/<i>"),
    and the outputs of calls ("<key>/<call>/input_ids", ".../attention_mask") whose keyword
    arguments, with the texts as indices and those of `from_pretrained` as `create`, are in the
    metadata "<key>/<call>", with the message of the error that transformers raises if it does.
    The metadata "<key>" have the tokenizer's class and settings, the ids of its special tokens,
    and the decoded ids of each text, by transformers ("text") and by tokenizer.json alone
    ("raw"; for skip_special_tokens also "raw_unnamed", of the ids without those of the
    tokenizer's named and extra special tokens, which transformers marks special). Copies the
    small JSON files of each repository to
    test/fixtures/processors/<org>--<repo>/[<subfolder>/]; the tokenizer_config.json of
    gemma-3-270m (1.1 MB) keeps only the `added_tokens_decoder` entries that the tests use."""
    from huggingface_hub import list_repo_files
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tensors = {}
    meta = {"texts": json.dumps(TOKENIZER_TEXTS), **versions()}
    keys = []
    for repo, revision, subfolder in TOKENIZER_REPOS:
        key = repo.replace("/", "--") + (f"--{subfolder}" if subfolder else "")
        keys.append(key)
        sub = dict(subfolder=subfolder) if subfolder else {}
        prefix = f"{subfolder}/" if subfolder else ""

        def load(**kwargs):
            return AutoTokenizer.from_pretrained(repo, revision=revision, **sub, **kwargs)

        tok = load()
        raw = Tokenizer.from_file(hf_hub_download(repo, prefix + "tokenizer.json",
                                                  revision=revision))
        used = set()
        decoded = {}
        named = set(tok.all_special_ids)
        for i, text in enumerate(TOKENIZER_TEXTS):
            ids = tok(text)["input_ids"]
            used.update(ids)
            tensors[f"{key}/ids/{i + 1}"] = torch.tensor(ids, dtype=torch.int32)
            tensors[f"{key}/content_ids/{i + 1}"] = torch.tensor(
                tok(text, add_special_tokens=False)["input_ids"], dtype=torch.int32)
            decoded[i + 1] = {str(skip): dict(text=tok.decode(ids, skip_special_tokens=skip),
                                              raw=raw.decode(ids, skip_special_tokens=skip))
                              for skip in (False, True)}
            decoded[i + 1]["True"]["raw_unnamed"] = raw.decode(
                [t for t in ids if t not in named], skip_special_tokens=True)
        specials = sorted({str(t) for t in tok.all_special_tokens} |
                          {str(t) for t in tok.special_tokens_map.values() if t})
        special_ids = {t: tok.convert_tokens_to_ids(t) for t in specials}
        used.update(special_ids.values())
        other_side = "left" if tok.padding_side == "right" else "right"
        pad = dict(pad_token=tok.eos_token) if tok.pad_token is None else {}
        calls = {
            "batch": (dict(text=[1, 2], padding=True), {}),
            "max_length": (dict(text=[1, 2], padding="max_length", max_length=12), {}),
            "other_side": (dict(text=[1, 2], padding=True, padding_side=other_side), {}),
            "truncate": (dict(text=[1, 3, 2], padding=True, truncation=True, max_length=6), {}),
            "truncate_only": (dict(text=[3], max_length=4), {}),
            "no_special": (dict(text=[1, 2], padding=True, add_special_tokens=False), {}),
            "single_padded": (dict(text=[1], padding=True), {}),
            "truncation_left": (dict(text=[3], truncation=True, max_length=4),
                                dict(truncation_side="left")),
            "padding_side": (dict(text=[1, 2], padding=True), dict(padding_side=other_side)),
            "no_bos": (dict(text=[1, 3], padding=True), dict(add_bos_token=False, **pad)),
            "eos": (dict(text=[1, 3], padding=True), dict(add_eos_token=True, **pad)),
            "bos_eos": (dict(text=[1, 3], padding=True, truncation=True, max_length=5),
                        dict(add_bos_token=True, add_eos_token=True, **pad)),
        }
        if pad:  # transformers can't pad without a padding token; setting one fixes it
            calls["pad_token"] = (dict(text=[1, 2], padding=True), pad)
        for name, (call, create) in calls.items():
            t = load(**create) if create else tok
            kwargs = {k: v for k, v in call.items() if k != "text"}
            record = dict(call, create=create)
            try:
                out = t([TOKENIZER_TEXTS[i - 1] for i in call["text"]], return_tensors="pt",
                        **kwargs)
                tensors[f"{key}/{name}/input_ids"] = out["input_ids"].int().contiguous()
                tensors[f"{key}/{name}/attention_mask"] = out["attention_mask"].int().contiguous()
                used.update(out["input_ids"].flatten().tolist())
            except ValueError as e:
                record["error"] = str(e)
            meta[f"{key}/{name}"] = json.dumps(record)
        max_length = tok.model_max_length
        meta[key] = json.dumps(dict(
            repo=repo, revision=revision, subfolder=subfolder, cls=type(tok).__name__,
            model_type=type(raw.model).__name__, settings=dict(bos_token=tok.bos_token, eos_token=tok.eos_token,
                          unk_token=tok.unk_token, pad_token=tok.pad_token,
                          sep_token=tok.sep_token, cls_token=tok.cls_token,
                          mask_token=tok.mask_token,
                          padding_side=tok.padding_side, truncation_side=tok.truncation_side,
                          model_max_length=max_length if max_length < 1e20 else None,
                          clean_up_tokenization_spaces=tok.clean_up_tokenization_spaces,
                          extra_special_tokens=[str(t) for t in tok.extra_special_tokens]),
            named_tokens={k: str(v) for k, v in tok.special_tokens_map.items()
                          if k not in type(tok).SPECIAL_TOKENS_ATTRIBUTES},
            special_ids=special_ids, decoded=decoded))
        if repo == SAM3_REPO:
            continue
        files = set(list_repo_files(repo, revision=revision))
        out = ROOT / "test" / "fixtures" / "processors" / repo.replace("/", "--") / subfolder
        out.mkdir(parents=True, exist_ok=True)
        for file in TOKENIZER_FILES:
            if prefix + file in files:
                shutil.copyfile(hf_hub_download(repo, prefix + file, revision=revision),
                                out / file)
        config = json.loads((out / "tokenizer_config.json").read_text())
        if len(config.get("added_tokens_decoder", {})) > 1000:
            config["added_tokens_decoder"] = {
                k: v for k, v in config["added_tokens_decoder"].items()
                if int(k) in used or v["content"] in special_ids}
            with open(out / "tokenizer_config.json", "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
    meta["keys"] = json.dumps(keys)
    path = ROOT / "test" / "references" / "processors" / "tokenizers.safetensors"
    save_file(tensors, path, metadata=meta)
    print(f"{path}: {path.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
