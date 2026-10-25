"""Write the fixtures and references of processors, through the full upstream path.

    uv run dump_processor.py [sam3] [qwen3_vl]    # all by default

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
    names = sys.argv[1:] or ["sam3", "qwen3_vl"]
    for name in names:
        {"qwen3_vl": qwen3_vl, "sam3": sam3}[name]()


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


if __name__ == "__main__":
    main()
