"""Write the fixtures and references of processors, through the full upstream path.

    uv run dump_processor.py [sam3] [qwen3_vl] [tokenizers]    # all by default

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

qwen3_vl: the processor of Qwen3.5 (`Qwen3VLProcessor`) as `AutoProcessor.from_pretrained(repo)`
loads it, called on tiny images and prompts with image placeholders, for
src/Transformers/processing_qwen3_vl.jl and image_processing_qwen2_vl.jl. Writes the small JSON
files of Qwen/Qwen3.5-0.8B and test/references/processors/qwen3_vl.safetensors, laid out as
SAM 3's (see `qwen3_vl`).

tokenizers: the tokenizers of the validated language models, of the FLUX.2 [klein] pipeline,
of Qwen3.5, of SAM 3 and of GPT-2, as `AutoTokenizer.from_pretrained(repo)` loads them, and
Qwen3.5's processor with keywords of calls in its tokenizer_config.json, for
src/Transformers/tokenization_utils_tokenizers.jl (see `tokenizers`).
"""
import json
import pathlib
import shutil
import sys
import tempfile
import types

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Sam3Processor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from common import versions

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIP = dict(image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711])


def main():
    names = sys.argv[1:] or ["sam3", "qwen3_vl", "tokenizers"]
    for name in names:
        {"qwen3_vl": qwen3_vl, "sam3": sam3, "tokenizers": tokenizers}[name]()


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


QWEN_REPO, QWEN_REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
QWEN_FILES = ["preprocessor_config.json", "tokenizer_config.json", "config.json",
              "video_preprocessor_config.json"]
# small patches and pixel bounds, so that the references stay small
QWEN_SMALL = dict(patch_size=4, size={"shortest_edge": 64, "longest_edge": 4096})
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


def qwen3_vl():
    """test/references/processors/qwen3_vl.safetensors: the outputs of the processor of
    Qwen/Qwen3.5-0.8B, `AutoProcessor.from_pretrained(repo, **create)(images=..., text=...)`,
    on tiny random images and prompts with image placeholders, as SAM 3's: "<name>/image/<i>",
    "<name>/<key>" and the token ids of each prompt before the expansion of its placeholders,
    "<name>/text_ids/<i>". The metadata of a case hold `create`, the prompts, the keywords of
    the call, the prompts with their placeholders expanded (`expanded`, as transformers'
    `get_text_with_replacements` writes them) and `legacy`, the keys added to
    preprocessor_config.json. The "default" case keeps the settings of the repository, and
    stores every `stride`-th patch of its `pixel_values` only. Also stores a table of
    `smart_resize` with the settings of the repository, and "tokenizer/...": the ids that the
    tokenizer of the repository (`Qwen3_5Tokenizer`, which rebuilds the normaliser and the
    pre-tokenizer of tokenizer.json in Python, and adds the tokens of tokenizer_config.json
    that tokenizer.json lacks) gives to varied strings, and their decoding."""
    copy_config_files(QWEN_REPO, QWEN_REVISION, QWEN_FILES)
    gen = torch.Generator().manual_seed(0)

    def uint8(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    tensors = {}
    meta = {"repo": QWEN_REPO, "revision": QWEN_REVISION, **versions()}
    tmp = tempfile.TemporaryDirectory()

    def legacy_dir(legacy):
        """The repository with `legacy` added to its preprocessor_config.json."""
        path = pathlib.Path(tmp.name) / "legacy"
        path.mkdir(exist_ok=True)
        for file in QWEN_FILES + ["tokenizer.json", "chat_template.jinja"]:
            shutil.copyfile(hf_hub_download(QWEN_REPO, file, revision=QWEN_REVISION),
                            path / file)
        config = json.loads((path / "preprocessor_config.json").read_text())
        (path / "preprocessor_config.json").write_text(json.dumps(dict(config, **legacy)))
        return str(path)

    def run(name, images, text=None, create=QWEN_SMALL, modes=None, call=None, legacy=None,
            stride=None):
        source = QWEN_REPO if legacy is None else legacy_dir(legacy)
        revision = QWEN_REVISION if legacy is None else None
        proc = AutoProcessor.from_pretrained(source, revision=revision, **create)
        pil = [Image.fromarray(img.numpy() if img.ndim == 2 else img.permute(1, 2, 0).numpy(),
                               mode=mode) if mode else img
               for img, mode in zip(images, modes or [None] * len(images))]
        out = proc(images=pil or None, text=text, return_tensors="pt", **(call or {}))
        for i, image in enumerate(images):
            tensors[f"{name}/image/{i + 1}"] = image.contiguous()
        for key, value in out.items():
            value = torch.as_tensor(value)
            if key == "pixel_values" and stride:
                value = value[::stride]
            tensors[f"{name}/{key}"] = value.contiguous()
        texts = [] if text is None else [text] if isinstance(text, str) else text
        for i, prompt in enumerate(texts):
            tensors[f"{name}/text_ids/{i + 1}"] = torch.tensor(proc.tokenizer(prompt)["input_ids"])
        expanded = texts
        if images and texts:
            replacements = [proc.replace_image_token(out, image_idx=i) for i in range(len(images))]
            expanded, _ = proc.get_text_with_replacements(list(texts), replacements)
        meta[name] = json.dumps(dict(create=create, text=text, call=call, modes=modes,
                                     legacy=legacy, stride=stride, expanded=expanded,
                                     cls=type(proc).__name__))
        return proc

    prompt = f"{IMAGE}Describe this image."
    run("exact", [uint8(3, 16, 24)], prompt)  # no resizing
    run("upscale", [uint8(3, 5, 7)], prompt)
    run("downscale", [uint8(3, 50, 70)], prompt,
        create=dict(QWEN_SMALL, size={"shortest_edge": 64, "longest_edge": 384}, **CLIP))
    run("bilinear", [uint8(3, 9, 13)], prompt, create=dict(QWEN_SMALL, resample=2))
    # images of reals in [0, 1], which Metalhead takes as already rescaled
    run("float", [torch.rand(3, 13, 21, generator=gen)], prompt,
        create=dict(QWEN_SMALL, do_rescale=False))
    # the 8-bit scale of an image of reals: Metalhead multiplies it by 255 without do_rescale
    x = torch.rand(3, 9, 13, generator=gen)
    run("float_no_rescale", [x * 255], prompt, create=dict(QWEN_SMALL, do_rescale=False))
    tensors["float_no_rescale/image/1"] = x
    run("grey", [uint8(9, 13)], prompt, modes=["L"])  # converted to RGB
    run("rgba", [uint8(4, 9, 13)], prompt, modes=["RGBA"])  # alpha dropped
    run("two_images", [uint8(3, 13, 21), uint8(3, 16, 24)],
        f"First {IMAGE}, then {IMAGE}: what changed?")
    # placeholders consumed in order through the prompts, padded to the longest
    run("batch", [uint8(3, 13, 21), uint8(3, 30, 8), uint8(3, 9, 13)],
        [f"{IMAGE}{IMAGE}Compare them.", "No image here.", f"Last: {IMAGE}"],
        call=dict(padding=True))
    run("no_resize", [uint8(3, 16, 24), uint8(3, 8, 32)], prompt * 2,
        create=dict(QWEN_SMALL, do_resize=False))
    run("no_rescale", [uint8(3, 9, 13)], prompt, create=dict(QWEN_SMALL, do_rescale=False))
    run("no_normalize", [uint8(3, 9, 13)], prompt, create=dict(QWEN_SMALL, do_normalize=False))
    run("neither", [uint8(3, 9, 13)], prompt,
        create=dict(QWEN_SMALL, do_rescale=False, do_normalize=False))
    # Qwen2-VL's legacy `min_pixels` and `max_pixels`, which override `size`
    run("legacy_file", [uint8(3, 50, 70)], prompt, create=dict(patch_size=4),
        legacy=dict(min_pixels=128, max_pixels=1024))
    run("legacy_create", [uint8(3, 50, 70)], prompt,
        create=dict(patch_size=4, min_pixels=128, max_pixels=2048))
    # without images, the placeholders stay; video tokens have the type 2
    run("text_only", [], "Hello <|image_pad|> and <|video_pad|> and <|audio_pad|>!")
    run("images_only", [uint8(3, 9, 13)])
    run("default", [uint8(3, 40, 56)], prompt, create={}, stride=40)

    proc = AutoProcessor.from_pretrained(QWEN_REPO, revision=QWEN_REVISION)
    p = proc.image_processor
    sizes = [(1, 1), (10, 10), (31, 33), (100, 3000), (255, 257), (256, 256), (480, 640),
             (375, 500), (1080, 1920), (3000, 4000), (4095, 4097), (5000, 7000), (8, 1500),
             (65, 12000)]
    factor = p.patch_size * p.merge_size
    table = [[h, w, *smart_resize(h, w, factor, p.size["shortest_edge"], p.size["longest_edge"])]
             for h, w in sizes]
    tensors["smart_resize"] = torch.tensor(table)
    meta["smart_resize"] = json.dumps(dict(factor=factor, min_pixels=p.size["shortest_edge"],
                                           max_pixels=p.size["longest_edge"]))
    qwen3_vl_tokenizer(proc.tokenizer, tensors, meta)

    out = ROOT / "test" / "references" / "processors" / "qwen3_vl.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {out.stat().st_size / 1e3:.1f} kB")


QWEN_TEXTS = [
    "Hello, world!",
    "I'm sure they'll say WE'RE fine, aren't they? DON'T",
    "Café naïve façade — “quotes” and ‘single’… São Paulo, R$ 12,50",
    "Café naïve, ñ and Å: decomposed accents",
    "你好，世界！今天天气很好。こんにちは、안녕하세요",
    "नमस्ते दुनिया, สวัสดีครับ, مرحبا بالعالم, Привет, Γειά σου",
    "Emoji 👋🏽🎉 and a family 👨‍👩‍👧‍👦 ok",
    "Pi is 3.14159265358979, and 1,000,000 > 999999; on 2024-10-25 at 12:34:56",
    "  leading and trailing  ",
    "a  b\t\tc\n\n\nd   \n  e\r\n\r\nwindows\r\nlines\t",
    "def f(x):\n    return x**2  # square\n",
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is it?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|image_pad|><|image_pad|><|video_pad|><|endoftext|>text<|im_end|>",
    # added tokens of tokenizer_config.json that tokenizer.json lacks
    "Audio <|audio_start|><|audio_pad|><|audio_end|> and <tts_pad><tts_text_bos_single>",
]


def qwen3_vl_tokenizer(tokenizer, tensors, meta):
    """The ids that `tokenizer` (`Qwen3_5Tokenizer`) gives to `QWEN_TEXTS`,
    "tokenizer/ids/<i>", and their decoding with and without the special tokens."""
    meta["tokenizer/class"] = type(tokenizer).__name__
    meta["tokenizer/texts"] = json.dumps(QWEN_TEXTS)
    decoded = []
    for i, text in enumerate(QWEN_TEXTS):
        ids = tokenizer(text)["input_ids"]
        tensors[f"tokenizer/ids/{i + 1}"] = torch.tensor(ids)
        decoded.append({str(skip): tokenizer.decode(ids, skip_special_tokens=skip)
                        for skip in (False, True)})
    meta["tokenizer/decoded"] = json.dumps(decoded)


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
    (QWEN_REPO, QWEN_REVISION, ""),
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
    tokenizer's named and extra special tokens, which transformers marks special). Under the
    key "Qwen3VLProcessor/<case>", the outputs of `AutoProcessor.from_pretrained` on the files
    of Qwen3.5 with keywords of calls in tokenizer_config.json ("init"), called on texts with
    keywords ("call"): `ProcessorMixin._merge_kwargs` makes the first the defaults of the
    second. Copies the small JSON files of each repository to
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
    processor_text_kwargs(tensors, meta)
    meta["keys"] = json.dumps(keys)
    path = ROOT / "test" / "references" / "processors" / "tokenizers.safetensors"
    save_file(tensors, path, metadata=meta)
    print(f"{path}: {path.stat().st_size / 1e3:.1f} kB")


# (keywords in tokenizer_config.json, keywords of the call, indices of the texts) of each case
PROCESSOR_TEXT_KWARGS = {
    "max_length": (dict(max_length=5), {}, [3, 6]),
    "call_max_length": (dict(max_length=5), dict(max_length=7), [3, 6]),
    "padding": (dict(padding="max_length", max_length=9), {}, [1, 2]),
    "truncation": (dict(truncation=True, max_length=6), dict(padding=True), [1, 3]),
}


def processor_text_kwargs(tensors, meta):
    """The cases of `tokenizers` "Qwen3VLProcessor/<case>"."""
    files = QWEN_FILES + ["tokenizer.json"]
    for name, (init, call, texts) in PROCESSOR_TEXT_KWARGS.items():
        with tempfile.TemporaryDirectory() as dir:
            for file in files:
                shutil.copyfile(hf_hub_download(QWEN_REPO, file, revision=QWEN_REVISION),
                                pathlib.Path(dir) / file)
            config = pathlib.Path(dir) / "tokenizer_config.json"
            config.write_text(json.dumps({**json.loads(config.read_text()), **init}))
            proc = AutoProcessor.from_pretrained(dir)
            out = proc(text=[TOKENIZER_TEXTS[i - 1] for i in texts], return_tensors="pt",
                       **call)
        key = f"Qwen3VLProcessor/{name}"
        for k in ("input_ids", "attention_mask", "mm_token_type_ids"):
            tensors[f"{key}/{k}"] = out[k].int().contiguous()
        meta[key] = json.dumps(dict(cls=type(proc).__name__, init=init, call=call, text=texts))


if __name__ == "__main__":
    main()
