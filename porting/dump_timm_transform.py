"""Write reference outputs of timm's eval transform and of Pillow's resize.

    uv run dump_timm_transform.py

Writes test/references/timm_transforms.safetensors, for `eval_transform` and `pil_resize` in
src/Timm/: the outputs of timm's full path,
`create_transform(**resolve_data_config({}, pretrained_cfg = cfg))`, applied to PIL images, for
`pretrained_cfg`s that cover the interpolations, crop modes and statistics of timm's tags, and
of `PIL.Image.resize` on tiny random 8-bit images. Images are stored (channels, height, width).

- "image/<i>": the small images, RGB but for a grayscale (1 channel) and an RGBA (4) one,
  which go through `convert("RGB")` first, as timm's datasets do.
- "transform/<case>": the `pixel_values` (batch, channels, height, width) of all the small
  images, or of the image of `large_image` for the case "realistic" (a timm model's own
  configuration); the metadata "transform/<case>" has the `pretrained_cfg`, `use_test_size` and
  the resolved data configuration.
- "resize/<i>/input" and "resize/<i>/<filter>": an image and `PIL.Image.resize` of it to the
  size in the metadata "resize/<i>", by each filter; "resize/<i>/torchvision_<filter>" its
  resize as a uint8 tensor by torchvision (`antialias=True`, on the CPU), for `pil_resize(...;
  torchvision = true)`, which transformers' image processors use.
"""
import json
import pathlib

import numpy as np
import PIL
import timm
import torch
from PIL import Image
from safetensors.torch import save_file
from timm.data import create_transform, resolve_data_config
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvF

from common import versions

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "test" / "references" / "timm_transforms.safetensors"

INCEPTION = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
CLIP = dict(mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
# (channels, height, width): landscape, odd portrait, smaller than the crops (upsampled), the
# size of a crop, two pixels high, grayscale and RGBA
SMALL = [(3, 20, 28), (3, 31, 17), (3, 9, 13), (3, 16, 16), (3, 2, 37), (1, 23, 19), (4, 25, 21)]
CASES = {
    "defaults": dict(input_size=(3, 16, 16)),
    "bilinear_095_inception": dict(input_size=(3, 16, 16), interpolation="bilinear",
                                   crop_pct=0.95, **INCEPTION),
    "bicubic_100_clip": dict(input_size=(3, 16, 16), interpolation="bicubic", crop_pct=1.0,
                             **CLIP),
    "pad_115": dict(input_size=(3, 16, 16), crop_pct=1.15),  # resized below the crop size
    "squash": dict(input_size=(3, 16, 16), crop_mode="squash", crop_pct=0.9),
    # sapiens2's settings: the only non-square input_size of timm's tags
    "squash_nonsquare": dict(input_size=(3, 20, 12), crop_mode="squash", crop_pct=1.0,
                             interpolation="bilinear"),
    # ResizeKeepRatio, which timm gives no interpolation: always bilinear
    "center_nonsquare": dict(input_size=(3, 12, 20), crop_pct=0.875, interpolation="bicubic"),
    "nearest": dict(input_size=(3, 16, 16), interpolation="nearest"),
    "box": dict(input_size=(3, 16, 16), interpolation="box", crop_pct=0.95),
    "hamming": dict(input_size=(3, 16, 16), interpolation="hamming"),
    "lanczos": dict(input_size=(3, 16, 16), interpolation="lanczos", crop_pct=1.0),
    # test_input_size and test_crop_pct only count with use_test_size = True
    "test_size_ignored": dict(input_size=(3, 16, 16), test_input_size=(3, 20, 20),
                              crop_pct=0.875, test_crop_pct=1.0),
    "test_size_used": dict(input_size=(3, 16, 16), test_input_size=(3, 20, 20),
                           crop_pct=0.875, test_crop_pct=1.0, use_test_size=True),
}
REALISTIC_MODEL = "convnext_tiny.fb_in1k"
LARGE = (3, 375, 500)
# (width, height) of the input and of the output; the first size is that of the input
RESIZES = [
    ((28, 20), (13, 9)),     # down, horizontal pass first
    ((13, 9), (28, 21)),     # up
    ((1, 1), (5, 3)),
    ((7, 1), (3, 4)),
    ((1, 9), (1, 4)),        # vertical pass only
    ((17, 31), (5, 31)),     # horizontal pass only
    ((40, 30), (35, 6)),     # the height shrinks most: vertical pass first
    ((30, 40), (7, 39)),
    ((64, 48), (11, 13)),    # by 5.8 and 3.7
    ((9, 6), (90, 2)),
]
FILTERS = {"nearest": Image.Resampling.NEAREST, "box": Image.Resampling.BOX,
           "bilinear": Image.Resampling.BILINEAR, "hamming": Image.Resampling.HAMMING,
           "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
TORCHVISION = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}


def large_image(c, h, w):
    """A deterministic 8-bit image with gradients, edges and noise, reproduced by
    `large_image` in test/processors/timm_transforms.jl."""
    c, h, w = np.meshgrid(np.arange(c), np.arange(h), np.arange(w), indexing="ij")
    v = (2 * h + 3 * w + 50 * c + (h * w * c + 7 * h + 11 * w) % 37 * 3
         + (h // 16 + w // 16) % 2 * 60) % 256
    return v.astype(np.uint8)


def to_pil(array):
    """A (channels, height, width) uint8 array as a PIL image, in RGB as timm reads images."""
    hwc = np.ascontiguousarray(array.transpose(1, 2, 0))
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[array.shape[0]]
    return Image.fromarray(hwc[:, :, 0] if mode == "L" else hwc, mode=mode)


def main():
    rng = np.random.default_rng(0)
    tensors, meta = {}, {"model": REALISTIC_MODEL, "pillow": PIL.__version__,
                         "cpu_capability": torch.backends.cpu.get_cpu_capability(), **versions()}

    small = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in SMALL]
    for i, image in enumerate(small):
        tensors[f"image/{i + 1}"] = torch.from_numpy(image)
    for name, cfg in CASES.items():
        cfg = dict(cfg)
        use_test_size = cfg.pop("use_test_size", False)
        config = resolve_data_config({}, pretrained_cfg=cfg, use_test_size=use_test_size)
        transform = create_transform(**config)
        x = torch.stack([transform(to_pil(image).convert("RGB")) for image in small])
        tensors[f"transform/{name}"] = x.contiguous()
        meta[f"transform/{name}"] = json.dumps(dict(pretrained_cfg=cfg,
                                                    use_test_size=use_test_size,
                                                    config=config))

    model = timm.create_model(REALISTIC_MODEL, pretrained=False)
    config = resolve_data_config({}, model=model)
    image = to_pil(large_image(*LARGE))
    tensors["transform/realistic"] = create_transform(**config)(image)[None].contiguous()
    meta["transform/realistic"] = json.dumps(dict(pretrained_cfg=model.pretrained_cfg,
                                                  use_test_size=False, config=config,
                                                  large_image=LARGE))

    for i, ((w, h), size) in enumerate(RESIZES):
        image = rng.integers(0, 256, (3, h, w), dtype=np.uint8)
        tensors[f"resize/{i + 1}/input"] = torch.from_numpy(image)
        for name, flt in FILTERS.items():
            out = np.asarray(to_pil(image).resize(size, flt)).transpose(2, 0, 1)
            tensors[f"resize/{i + 1}/{name}"] = torch.from_numpy(np.ascontiguousarray(out))
        for name, mode in TORCHVISION.items():
            out = tvF.resize(torch.from_numpy(image), [size[1], size[0]], interpolation=mode,
                             antialias=True)
            tensors[f"resize/{i + 1}/torchvision_{name}"] = out.clone().contiguous()
        meta[f"resize/{i + 1}"] = json.dumps(dict(size=size))

    save_file(tensors, OUT, metadata=meta)
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
