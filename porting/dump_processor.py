"""Write the fixtures and references of processors, through the full upstream path.

    uv run dump_processor.py [sam3] [tokenizers]    # all by default

sam3: the processor of SAM 3 as `AutoProcessor.from_pretrained(repo)` loads it, called on tiny
images, prompts and boxes (`proc(images=..., text=..., input_boxes=...)`), for
src/Transformers/processing_sam3.jl. Writes

- test/fixtures/processors/<org>--<repo>/: the small JSON files of unsloth/sam3.1 and
  jetjodh/sam3 (not tokenizer.json), from which the Julia tests load the processor offline;
- test/references/processors/sam3.safetensors: for each case `<name>`, its images
  "<name>/image/<i>" (channels, height, width; or height, width for a grey image), the
  processor's outputs "<name>/<key>", and the token ids of each prompt "<name>/text_ids/<i>",
  which the tests pass in place of the strings without a tokenizer backend. The metadata hold
  each case's settings (`create`, the keyword arguments of `from_pretrained`), prompts and boxes.
  "tokenizer/..." cases call the tokenizer alone (padding, truncation, decoding), and "post/..."
  the post-processing of random model outputs.

tokenizers: the tokenizers of the validated language models, of the FLUX.2 [klein] pipeline,
of Qwen3.5, of SAM 3 and of GPT-2, as `AutoTokenizer.from_pretrained(repo)` loads them, for
src/Transformers/tokenization_utils_tokenizers.jl (see `tokenizers`).
"""
import json
import pathlib
import shutil
import sys
import types

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Sam3Processor

from common import versions

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIP = dict(image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711])


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
SAM3_VIDEO_REPO, SAM3_VIDEO_REVISION = "jetjodh/sam3", "1aa50ce07302cb375f85d8084b68a0fb378b8d85"
SAM3_FILES = ["processor_config.json", "tokenizer_config.json", "special_tokens_map.json"]


def sam3():
    for repo, revision in [(SAM3_REPO, SAM3_REVISION), (SAM3_VIDEO_REPO, SAM3_VIDEO_REVISION)]:
        copy_config_files(repo, revision, SAM3_FILES)

    gen = torch.Generator().manual_seed(0)

    def uint8(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    small = dict(size={"height": 16, "width": 24})
    tensors = {}
    meta = {"repo": SAM3_REPO, "revision": SAM3_REVISION, **versions()}

    def run(name, images, create=small, text=None, boxes=None, labels=None, modes=None,
            repo=SAM3_REPO, revision=SAM3_REVISION, cls=AutoProcessor):
        """`cls.from_pretrained(repo, **create)(images=..., text=..., input_boxes=...)`;
        `modes` gives the PIL mode of each image ("L", "RGBA"), RGB by default."""
        proc = cls.from_pretrained(repo, revision=revision, **create)
        pil = [Image.fromarray(img.numpy() if img.ndim == 2 else img.permute(1, 2, 0).numpy(),
                               mode=mode) if mode else img
               for img, mode in zip(images, modes or [None] * len(images))]
        kwargs = {k: v for k, v in dict(text=text, input_boxes=boxes,
                                        input_boxes_labels=labels).items() if v is not None}
        out = proc(images=pil, return_tensors="pt", **kwargs)
        for i, image in enumerate(images):
            tensors[f"{name}/image/{i + 1}"] = image.contiguous()
        for key, value in out.items():
            tensors[f"{name}/{key}"] = torch.as_tensor(value).contiguous()
        prompts = text if isinstance(text, list) else [text]
        if text is None and boxes is not None:
            prompts = ["visual"]
        for i, prompt in enumerate(prompts):
            if prompt is None and boxes is not None:
                prompt = "visual"
            if prompt is not None:
                ids = proc.tokenizer(prompt)["input_ids"]
                tensors[f"{name}/text_ids/{i + 1}"] = torch.tensor(ids)
        meta[name] = json.dumps(dict(create=create, text=text, boxes=boxes, labels=labels,
                                     modes=modes, repo=repo, revision=revision,
                                     cls=type(proc).__name__))
        return proc

    proc = run("exact", [uint8(3, 16, 24)], text="a cat")  # no resizing
    run("upscale", [uint8(3, 9, 13)], text="a cat")
    run("downscale", [uint8(3, 50, 70)], create=dict(small, **CLIP), text=["a dog"])
    # images of reals in [0, 1], which Metalhead takes as already rescaled
    run("float", [torch.rand(3, 9, 13, generator=gen)], create=dict(small, do_rescale=False),
        text="a cat")
    run("batch", [uint8(3, 9, 13), uint8(3, 16, 24), uint8(3, 30, 8)],
        text=["a cat", "two dogs on a sofa", "a red car"],
        boxes=[[[1, 2, 5, 7]], [[0, 0, 20, 10], [3.5, 4, 12, 15.25]], [[2, 3, 6, 25]]],
        labels=[[1], [0, 1], [1]])
    run("boxes_only", [uint8(3, 9, 13)], boxes=[[[1, 2, 5, 7], [0, 0, 13, 9]]])
    run("boxes_none", [uint8(3, 9, 13), uint8(3, 9, 13)], text=["a cat", None],
        boxes=[None, [[1, 2, 5, 7]]])
    run("bicubic", [uint8(3, 9, 13)], create=dict(small, resample=3), text="a cat")
    run("no_resize", [uint8(3, 9, 13), uint8(3, 9, 13)], create=dict(do_resize=False),
        text=["a cat", "a dog"])
    run("no_rescale", [uint8(3, 9, 13)], create=dict(small, do_rescale=False), text="a cat")
    run("no_normalize", [uint8(3, 9, 13)], create=dict(small, do_normalize=False),
        text="a cat")
    run("neither", [uint8(3, 9, 13)], create=dict(small, do_rescale=False, do_normalize=False),
        text="a cat")
    # the 8-bit scale of an image of reals: Metalhead multiplies it by 255 without do_rescale
    x = torch.rand(3, 9, 13, generator=gen)
    run("float_no_rescale", [x * 255], create=dict(small, do_rescale=False), text="a cat")
    tensors["float_no_rescale/image/1"] = x
    run("grey", [uint8(9, 13)], text="a cat", modes=["L"])  # converted to RGB
    run("rgba", [uint8(4, 9, 13)], text="a cat", modes=["RGBA"])  # alpha dropped
    long = "a photo of " + " and ".join(["a small red cat"] * 8)
    run("long", [uint8(3, 9, 13)], text=long)  # more than 32 tokens: not truncated

    # jetjodh/sam3 names Sam3VideoProcessor, which takes no prompts: its images, and the
    # whole processor as Sam3Processor loads it
    run("video_processor", [uint8(3, 9, 13)], repo=SAM3_VIDEO_REPO,
        revision=SAM3_VIDEO_REVISION)
    run("video_sam3", [uint8(3, 9, 13)], text="a cat", boxes=[[[1, 2, 5, 7]]],
        repo=SAM3_VIDEO_REPO, revision=SAM3_VIDEO_REVISION, cls=Sam3Processor)

    sam3_tokenizer(proc.tokenizer, tensors, meta, long)
    sam3_post_processing(proc, tensors, meta)

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


def sam3_post_processing(proc, tensors, meta):
    """The post-processing of random outputs of SAM 3, of 2 images and 5 queries, as
    "post/outputs/<key>" and the results of each image "post/<method>/<i>/<key>"."""
    gen = torch.Generator().manual_seed(1)
    b, q, h, w = 2, 5, 8, 6
    boxes = torch.rand(b, q, 4, generator=gen)
    boxes = torch.cat([boxes[..., :2].minimum(boxes[..., 2:]), boxes[..., :2].maximum(boxes[..., 2:])],
                      dim=-1)
    outputs = types.SimpleNamespace(
        pred_logits=torch.randn(b, q, generator=gen) * 2,
        pred_boxes=boxes,
        presence_logits=torch.tensor([[2.0], [0.5]]),
        pred_masks=torch.randn(b, q, h, w, generator=gen) * 3,
        semantic_seg=torch.randn(b, 1, h, w, generator=gen) * 3)
    for key, value in vars(outputs).items():
        tensors[f"post/outputs/{key}"] = value.contiguous()
    sizes = [(12, 16), (9, 7)]
    results = {
        "detection": proc.post_process_object_detection(outputs, threshold=0.3,
                                                        target_sizes=sizes),
        "detection_normalized": proc.post_process_object_detection(outputs, threshold=0.3),
        "instance": proc.post_process_instance_segmentation(
            outputs, threshold=0.3, mask_threshold=0.5, target_sizes=sizes),
        "instance_normalized": proc.post_process_instance_segmentation(outputs, threshold=0.3),
        "semantic": proc.post_process_semantic_segmentation(
            outputs, target_sizes=sizes, threshold=0.5, return_segmentation_scores=True),
        "semantic_normalized": proc.post_process_semantic_segmentation(
            outputs, threshold=0.5, return_segmentation_scores=True),
    }
    for name, per_image in results.items():
        for i, result in enumerate(per_image):
            for key, value in result.items():
                tensors[f"post/{name}/{i + 1}/{key}"] = value.contiguous()
    meta["post"] = json.dumps(dict(target_sizes=sizes, threshold=0.3, mask_threshold=0.5,
                                   semantic_threshold=0.5))


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
