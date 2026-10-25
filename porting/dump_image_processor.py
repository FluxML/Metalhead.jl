"""Write reference outputs of the image processors of Qwen3.5 and SAM 3.

    uv run dump_image_processor.py [qwen2_vl] [sam3]    # both by default

Writes test/references/qwen2_vl_image_processor.safetensors, for `qwen3_5_image_inputs` in
src/Transformers/image_processing_qwen2_vl.jl: the `pixel_values` and `image_grid_thw` that
the image processor of Qwen/Qwen3.5-0.8B (transformers' `Qwen2VLImageProcessor`) makes of tiny
random images, 8-bit or float, with small patches and pixel bounds so that the file stays small,
and a table of `smart_resize` with the checkpoint's own settings. Each case `<name>` stores
"<name>/image/<i>" (channels, height, width), "<name>/pixel_values" and "<name>/grid_thw",
and the processor's keyword arguments in the metadata.
"""
import json
import pathlib
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from common import versions

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
SMALL = dict(patch_size=4, merge_size=2, temporal_patch_size=2,
             size={"shortest_edge": 64, "longest_edge": 4096})
CLIP = dict(image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711])


def main():
    names = sys.argv[1:] or ["qwen2_vl", "sam3"]
    for name in names:
        {"qwen2_vl": qwen2_vl, "sam3": sam3}[name]()


def qwen2_vl():
    processor = AutoImageProcessor.from_pretrained(REPO, revision=REVISION)
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
    tensors, meta = {}, {"repo": REPO, "revision": REVISION, "class": type(processor).__name__,
                         **versions()}
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


SAM3_REPO, SAM3_REVISION = "jetjodh/sam3", "1aa50ce07302cb375f85d8084b68a0fb378b8d85"


def sam3():
    """test/references/sam3_image_processor.safetensors, for `sam3_image_inputs` in
    src/Transformers/image_processing_sam3.jl: the `pixel_values` that the image processor of
    SAM 3 (`Sam3ImageProcessor`) makes of tiny random images, 8-bit or float, resized to a
    small `size`. Each case `<name>` stores "<name>/image/<i>" (channels, height, width) and
    "<name>/pixel_values", and the processor's keyword arguments in the metadata."""
    from transformers import Sam3Processor
    processor = Sam3Processor.from_pretrained(SAM3_REPO, revision=SAM3_REVISION).image_processor
    gen = torch.Generator().manual_seed(0)

    def uint8(h, w):
        return torch.randint(0, 256, (3, h, w), generator=gen, dtype=torch.uint8)

    small = dict(size={"height": 16, "width": 24})
    cases = {
        "exact": ([uint8(16, 24)], small),  # no resizing
        "upscale": ([uint8(9, 13)], small),
        "downscale": ([uint8(50, 70)], dict(small, **CLIP)),
        "float": ([torch.rand(3, 9, 13, generator=gen)], dict(small, do_rescale=False)),
        "batch": ([uint8(9, 13), uint8(16, 24), uint8(30, 8)], small),
    }
    tensors, meta = {}, {"repo": SAM3_REPO, "revision": SAM3_REVISION,
                         "class": type(processor).__name__, **versions()}
    for name, (images, kwargs) in cases.items():
        out = processor(images=images, return_tensors="pt", **kwargs)
        for i, image in enumerate(images):
            tensors[f"{name}/image/{i + 1}"] = image.contiguous()
        tensors[f"{name}/pixel_values"] = out["pixel_values"].contiguous()
        meta[name] = json.dumps(kwargs)
    out = ROOT / "test" / "references" / "sam3_image_processor.safetensors"
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {meta['class']}, {out.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
