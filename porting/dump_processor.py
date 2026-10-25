"""Write the fixtures and references of processors, through the full upstream path.

    uv run dump_processor.py [sam3] [qwen2_vl]    # all by default

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

qwen2_vl: test/references/qwen2_vl_image_processor.safetensors, the image processor of Qwen3.5
(see `qwen2_vl`).
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
from transformers import AutoImageProcessor, AutoProcessor, Sam3Processor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from common import versions

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIP = dict(image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711])


def main():
    names = sys.argv[1:] or ["sam3", "qwen2_vl"]
    for name in names:
        {"qwen2_vl": qwen2_vl, "sam3": sam3}[name]()


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
SMALL = dict(patch_size=4, merge_size=2, temporal_patch_size=2,
             size={"shortest_edge": 64, "longest_edge": 4096})


def qwen2_vl():
    """test/references/qwen2_vl_image_processor.safetensors, for `qwen3_5_image_inputs` in
    src/Transformers/image_processing_qwen2_vl.jl: the `pixel_values` and `image_grid_thw` that
    the image processor of Qwen/Qwen3.5-0.8B (transformers' `Qwen2VLImageProcessor`) makes of
    tiny random images, 8-bit or float, with small patches and pixel bounds so that the file
    stays small, and a table of `smart_resize` with the checkpoint's own settings. Each case
    `<name>` stores "<name>/image/<i>" (channels, height, width), "<name>/pixel_values" and
    "<name>/grid_thw", and the processor's keyword arguments in the metadata."""
    processor = AutoImageProcessor.from_pretrained(QWEN_REPO, revision=QWEN_REVISION)
    gen = torch.Generator().manual_seed(0)

    def uint8(h, w):
        return torch.randint(0, 256, (3, h, w), generator=gen, dtype=torch.uint8)

    cases = {
        "exact": ([uint8(16, 24)], SMALL),  # no resizing
        "upscale": ([uint8(13, 21)], SMALL),
        "downscale": ([uint8(50, 70)], dict(SMALL, size={"shortest_edge": 64,
                                                         "longest_edge": 384}, **CLIP)),
        "float": ([torch.rand(3, 13, 21, generator=gen)], dict(SMALL, do_rescale=False)),
        "batch": ([uint8(13, 21), uint8(16, 24), uint8(8, 8)], SMALL),
    }
    tensors, meta = {}, {"repo": QWEN_REPO, "revision": QWEN_REVISION,
                         "class": type(processor).__name__, **versions()}
    for name, (images, kwargs) in cases.items():
        out = processor(images=images, return_tensors="pt", **kwargs)
        for i, image in enumerate(images):
            tensors[f"{name}/image/{i + 1}"] = image.contiguous()
        tensors[f"{name}/pixel_values"] = out["pixel_values"].contiguous()
        tensors[f"{name}/grid_thw"] = out["image_grid_thw"].contiguous()
        meta[name] = json.dumps(kwargs)

    sizes = [(1, 1), (10, 10), (31, 33), (100, 3000), (255, 257), (256, 256), (480, 640),
             (375, 500), (1080, 1920), (3000, 4000), (4095, 4097), (5000, 7000), (8, 1500),
             (65, 12000)]
    p = processor
    factor = p.patch_size * p.merge_size
    table = [[h, w, *smart_resize(h, w, factor, p.size["shortest_edge"], p.size["longest_edge"])]
             for h, w in sizes]
    tensors["smart_resize"] = torch.tensor(table)
    meta["smart_resize"] = json.dumps(dict(factor=factor, min_pixels=p.size["shortest_edge"],
                                           max_pixels=p.size["longest_edge"]))

    out = ROOT / "test" / "references" / "qwen2_vl_image_processor.safetensors"
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {meta['class']}, {out.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
