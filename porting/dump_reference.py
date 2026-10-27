"""Write reference outputs of a real pretrained checkpoint.

    uv run dump_reference.py timm/<model> [<org>/<model> ...] [--revision SHA]

Writes test/references/<org>--<model>.safetensors for each repo, with the logits and
pre-logits features of the checkpoint on `reference_input`, computed in float32. The metadata records the repo and
the commit sha, which the Julia test pins.

For a transformers causal LM, the input is `PROMPTS` tokenized by the repo's tokenizer and cut to
a common length (`input_ids`, 0-based), and the outputs are the hidden states after the final
norm, the logits of the last position, and the greedy continuation of the first prompt
(`generated_ids`), in float32, so that the Julia tests need neither Python nor a tokenizer.
With `--layers N`, the model keeps its first N decoder layers. `--selective-shards` also
downloads only the shards of the kept weights (see `first_layers`), for a sharded checkpoint
too large to download whole (Qwen3-30B-A3B); its omitted layers are not validated.

For a text encoder (BERT, XLM-RoBERTa, ModernBERT, with or without a head), the input is
`ENCODER_TEXTS` tokenized by the repo's tokenizer and padded as a batch (`input_ids`,
`attention_mask` and `token_type_ids` where it returns them), or, for a classifier,
`ENCODER_PAIRS`, which the tokenizer joins as a (query, passage) pair; the outputs are the
hidden states of the base model, its `pooler_output` if it has one, the logits of a
classifier and, for a masked language model, those of the last token of each text
(`last_logits`). For a sentence-transformers repo, `--encoder` dumps these instead of its
embeddings; both are kept in the file of the repo (`save_reference`).

A diffusers component is named `<org>/<repo>/<subfolder>` (e.g. `.../vae`): an autoencoder
encodes and decodes `reference_input` images, and a FLUX.2 or Z-Image transformer predicts the
flow of random latents; the metadata has its parameter count. A diffusers pipeline
(`<org>/<repo>` with a model_index.json) generates an image from `PIPELINE_PROMPT` end to end,
from fixed initial noise. FLUX.2 [klein] also edits the image of `IMAGE_URL` as `EDIT_PROMPT`
says. Random inputs are stored rounded to float16, so that both sides use the same values.

A sentence-transformers model (`<org>/<repo>` with a modules.json) is run through its `encode`,
`encode_query` and `encode_document` on `EMBEDDING_CALLS`, in float32: the embeddings, the inputs
of the model and the similarities of the queries and the documents.

A CLIP model (`CLIPModel`, or a `CLIPTextModel` such as the `text_encoder` subfolder of a
Stable Diffusion pipeline) embeds `CLIP_TEXTS` and two images, through the repository's
processor (see `dump_clip_reference`).
"""
import argparse
import contextlib
import json
import re
import tempfile

import numpy as np
import pathlib

import timm
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from common import reference_input, versions

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repos", nargs="+", metavar="repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--layers", type=int,
                   help="a language model cut to its first N decoder layers, for checkpoints "
                        "too large for the machine (see `first_layers`)")
    p.add_argument("--encoder", action="store_true",
                   help="for a sentence-transformers repo, dump the outputs of its BERT, "
                        "XLM-RoBERTa or ModernBERT encoder (`dump_encoder_reference`) rather "
                        "than its embeddings; both go in the same file")
    p.add_argument("--selective-shards", action="store_true",
                   help="with --layers, download only kept-weight shards; skip omitted-layer validation")
    args = p.parse_args()
    if args.selective_shards and args.layers is None:
        p.error("--selective-shards needs --layers")
    for repo in args.repos:
        dump_reference(repo, args.revision, args.batch, args.layers, args.selective_shards,
                       args.encoder)


PROMPTS = ["The capital of France is Paris, and the capital of Italy is",
           "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]


def dump_reference(repo, revision, batch, layers=None, selective_shards=False, encoder=False):
    parts = repo.split("/")
    repo, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    sha = HfApi().model_info(repo, revision=revision).sha
    if subfolder:
        config = json.loads(pathlib.Path(hf_hub_download(
            repo, f"{subfolder}/config.json", revision=sha)).read_text())
        if config.get("model_type", "").startswith("clip"):  # a transformers component
            return dump_clip_reference(repo, sha, config, subfolder)
        return dump_diffusers_reference(repo, subfolder, sha, batch)
    siblings = {f.rfilename for f in HfApi().model_info(repo, revision=sha).siblings}
    if "model_index.json" in siblings:
        return dump_pipeline_reference(repo, sha)
    if "modules.json" in siblings and not encoder:
        return dump_sentence_transformers_reference(repo, sha)
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    if config.get("model_type") in ("sam3", "sam3_video"):
        return dump_sam3_reference(repo, sha)
    if config.get("model_type") in ("siglip", "siglip2"):
        return dump_siglip_reference(repo, sha)
    if config.get("model_type", "").startswith("clip"):
        return dump_clip_reference(repo, sha, config)
    if config.get("model_type") in ("bert", "xlm-roberta", "modernbert"):
        return dump_encoder_reference(repo, sha)
    if "model_type" in config:
        return dump_lm_reference(repo, sha, layers=layers, selective_shards=selective_shards)
    if layers is not None:
        raise SystemExit("--layers needs a transformers language model")
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


def dump_lm_reference(repo, sha, max_new_tokens=8, layers=None, selective_shards=False):
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer)
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    # a vision-language model: AutoModelForCausalLM would build its text model only
    auto = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    if selective_shards and layers is None:
        raise ValueError("--selective-shards needs --layers")
    kwargs = {}
    if layers is not None and not selective_shards:
        kwargs["config"] = truncated_config(AutoConfig.from_pretrained(repo, revision=sha), layers)
    source = first_layers(repo, sha, layers) if selective_shards else contextlib.nullcontext((repo, sha))
    with source as (source, revision):
        model = auto.from_pretrained(source, revision=revision, dtype=torch.float32,
                                     attn_implementation="sdpa",
                                     experts_implementation="eager", **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha)
    ids = [tokenizer(p)["input_ids"] for p in PROMPTS]
    n = min(map(len, ids))
    input_ids = torch.tensor([i[:n] for i in ids])
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
        # the head of the model, which may scale (Granite) or softcap (Gemma) the logits
        logits = model(input_ids=input_ids, logits_to_keep=1).logits[:, -1, :]
        generated = model.generate(input_ids[:1], max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)

    out = ROOT / "test" / "references" / (repo.replace("/", "--") + ".safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"repo": repo, "revision": sha, "prompts": json.dumps(PROMPTS), **versions()}
    if layers is not None:
        meta["num_hidden_layers"] = str(layers)
        if selective_shards:
            meta["selective_shards"] = "true"
    tensors = {"input_ids": input_ids, "hidden_states": hidden.contiguous(),
               "last_logits": logits.contiguous(), "generated_ids": generated.contiguous()}
    if "vision_config" in config:
        image_tensors, meta["image_text"] = image_reference(model, repo, sha, max_new_tokens)
        tensors.update(image_tensors)
        meta["image_prompt"] = IMAGE_PROMPT
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB, greedy continuation "
          f"{tokenizer.decode(generated[0, n:])!r}")


def truncated_config(config, layers):
    """`config` with the first `layers` decoder layers (of its `text_config`, if present)."""
    text = getattr(config, "text_config", config)
    if not 0 < layers <= text.num_hidden_layers:
        raise ValueError(f"--layers must be in 1:{text.num_hidden_layers}, got {layers}")
    if getattr(text, "layer_types", None) is not None:
        text.layer_types = text.layer_types[:layers]
    text.num_hidden_layers = layers
    return config


DECODER_LAYER = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")


@contextlib.contextmanager
def first_layers(repo, sha, num_hidden_layers):
    """The source of `from_pretrained` for the checkpoint `repo` cut to its first
    `num_hidden_layers` decoder layers, and its revision: with `None`, the repository; else a
    temporary directory with its config.json, whose text model has that many layers, and
    links to its other small files and to the shards of the weights kept, those of the first
    layers, the embeddings, the final norm, the head and the vision encoder, which alone are
    downloaded, and their index. The multi-token prediction weights (`mtp.*`), which
    transformers doesn't load, are left out. Metalhead's tests load the same weights
    (`reference_model` in test/parity.jl)."""
    if num_hidden_layers is None:
        yield repo, sha
        return
    from huggingface_hub import HfApi
    n = num_hidden_layers
    index = json.loads(pathlib.Path(
        hf_hub_download(repo, "model.safetensors.index.json", revision=sha)).read_text())

    def kept(key):
        m = DECODER_LAYER.match(key)
        return int(m[1]) < n if m else not key.startswith("mtp.")

    weight_map = {k: v for k, v in index["weight_map"].items() if kept(k)}
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    text = config.get("text_config", config)
    if not 0 < n <= text["num_hidden_layers"]:
        raise ValueError(f"--layers must be in 1:{text['num_hidden_layers']}, got {n}")
    text["num_hidden_layers"] = n
    if text.get("layer_types"):
        text["layer_types"] = text["layer_types"][:n]
    small = [f.rfilename for f in HfApi().model_info(repo, revision=sha).siblings
             if f.rfilename.endswith((".json", ".txt", ".jinja"))
             and f.rfilename not in ("config.json", "model.safetensors.index.json")]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for f in small + sorted(set(weight_map.values())):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).symlink_to(hf_hub_download(repo, f, revision=sha))
        (tmp / "config.json").write_text(json.dumps(config, indent=2))
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map}))
        yield str(tmp), None


EMBEDDING_CALLS = [
    ("encode_query", ["What is the capital of China?", "Explain gravity"]),
    ("encode_document", ["The capital of China is Beijing.",
                         "Gravity is a force that attracts two bodies towards each other. It "
                         "gives weight to physical objects and is responsible for the movement "
                         "of planets around the sun."]),
    ("encode", ["A man is eating food.", "Un homme mange de la nourriture.",
                "The quick brown fox jumps over the lazy dog, " * 3])]


def dump_sentence_transformers_reference(repo, sha):
    """The embeddings of a sentence-transformers model, in float32, by each method of
    `EMBEDDING_CALLS` on its texts, in one batch ("<method>/embeddings"), with the inputs of
    the model ("<method>/input_ids", 0-based, "attention_mask"), and the similarities of the
    queries and the documents ("similarity")."""
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(repo, revision=sha, device="cpu",
                                model_kwargs={"dtype": torch.float32,
                                              "attn_implementation": "sdpa"})
    tensors, embeddings = {}, {}
    with torch.no_grad():
        for method, texts in EMBEDDING_CALLS:
            e = getattr(model, method)(texts, convert_to_tensor=True)
            prompt = {"encode_query": model.prompts.get("query"),
                      "encode_document": model.prompts.get("document")}.get(method)
            features = model.preprocess(texts, prompt=prompt or None)
            tensors.update({f"{method}/embeddings": e,
                            f"{method}/input_ids": features["input_ids"],
                            f"{method}/attention_mask": features["attention_mask"]})
            embeddings[method] = e
        tensors["similarity"] = model.similarity(embeddings["encode_query"],
                                                 embeddings["encode_document"])
    out = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "calls": json.dumps(EMBEDDING_CALLS),
            "sentence_transformers": sentence_transformers.__version__, **versions()}
    save_reference(out, tensors, meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB, similarities "
          f"{tensors['similarity'].tolist()}")


IMAGE_PROMPT = "Describe this image in one sentence."
IMAGE_URL = ("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
             "pipeline-cat-chonk.jpeg")


def image_reference(model, repo, sha, max_new_tokens, size=(224, 160)):
    """The outputs of a vision-language model on `IMAGE_PROMPT` about the image of `IMAGE_URL`,
    resized to `size` (w, h) and preprocessed by the repo's processor with `min_pixels` low
    enough to keep its size, under keys "image/...", and the text of the chat template. The pixel
    values are rounded to float16; "image/image" is the resized image, (3, h, w) in uint8."""
    import gc
    import io
    import urllib.request
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(repo, revision=sha)
    with urllib.request.urlopen(IMAGE_URL) as f:
        image = Image.open(io.BytesIO(f.read())).convert("RGB").resize(size, Image.BICUBIC)
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": IMAGE_PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt",
                       images_kwargs={"min_pixels": size[0] * size[1]})
    inputs["pixel_values"] = inputs["pixel_values"].half().float()
    with torch.no_grad():
        hidden = model.model(**inputs).last_hidden_state
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n = inputs["input_ids"].shape[1]
    print(f"{repo} on the image: {processor.decode(generated[0, n:])!r}")
    return {"image/image": torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous(),
            "image/input_ids": inputs["input_ids"],
            "image/pixel_values": inputs["pixel_values"].half(),
            "image/image_grid_thw": inputs["image_grid_thw"],
            "image/hidden_states": hidden.contiguous(), "image/last_logits": logits.contiguous(),
            "image/generated_ids": generated.contiguous()}, text


def dump_sam3_reference(repo, sha, size=336, top=2, stride=2):
    """The detections of SAM 3 (the detector, `Sam3Model`) in the image of `IMAGE_URL`,
    resized to `size` × `size` and stored as float16 values in [0, 1] ("image"), which the
    processor upsamples to the model's size, from the same vision features: "text/..." for the
    text prompt "cat", "box/..." for the text "visual" (the processor's text of box prompts)
    with a positive box, the first box found for "cat", in the pixels of the image
    ("box/boxes_xyxy"). Each stores the processor's inputs, the outputs of the model but its
    masks, of which only those of the `top` best queries ("mask_queries", 1-based), every
    `stride` pixels; and the scores, boxes and mask areas of transformers'
    `post_process_instance_segmentation` at the size of the image ("post/...")."""
    import gc
    import io
    import urllib.request
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(repo, revision=sha, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
    processor = Sam3Processor.from_pretrained(repo, revision=sha)
    with urllib.request.urlopen(IMAGE_URL) as f:
        image = Image.open(io.BytesIO(f.read())).convert("RGB").resize((size, size), Image.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).half()
    image = pixels.float()
    pixel_values = processor(images=[image], do_rescale=False, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        vision = model.get_vision_features(pixel_values=pixel_values)

    tensors, summary = {"image": pixels}, {}

    def run(name, **kwargs):
        inputs = processor(images=[image], do_rescale=False, return_tensors="pt", **kwargs)
        inputs.pop("pixel_values")
        inputs.pop("original_sizes")
        with torch.no_grad():
            outputs = model(vision_embeds=vision, **inputs)
        scores = outputs.pred_logits.sigmoid() * outputs.presence_logits.sigmoid()
        queries = scores.topk(top, dim=1).indices[0]
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[(size, size)])[0]
        tensors.update({f"{name}/{k}": v for k, v in inputs.items()})
        tensors.update({
            f"{name}/pred_logits": outputs.pred_logits, f"{name}/pred_boxes": outputs.pred_boxes,
            f"{name}/presence_logits": outputs.presence_logits,
            f"{name}/mask_queries": queries + 1,
            f"{name}/pred_masks": outputs.pred_masks[:, queries, ::stride, ::stride],
            f"{name}/semantic_seg": outputs.semantic_seg[:, :, ::stride, ::stride],
            f"{name}/post/scores": results["scores"], f"{name}/post/boxes": results["boxes"],
            f"{name}/post/mask_areas": results["masks"].sum(dim=(1, 2))})
        summary[name] = (round(outputs.presence_logits.sigmoid().item(), 4), len(results["scores"]))
        return outputs

    first = run("text", text="cat")
    box = (first.pred_boxes[0, first.pred_logits[0].argmax()] * size).tolist()
    run("box", text="visual", input_boxes=[[box]])
    tensors["box/boxes_xyxy"] = torch.tensor([box])
    out = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "stride": str(stride), **versions()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB, "
          f"(presence, objects) {summary}")


SIGLIP_TEXTS = ["a photo of 2 cats", "a photo of 2 dogs", "Two CATS asleep on a pink couch"]
COCO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def dump_siglip_reference(repo, sha, size=(224, 168), stride=8):
    """The similarities of SigLIP or SigLIP 2 (`SiglipModel`, `Siglip2Model`) between the
    image of `COCO_URL`, resized to `size` (w, h) and stored as uint8 ("image"), and
    `SIGLIP_TEXTS`, from the inputs of the repo's processor (`padding="max_length"` for a
    `SiglipProcessor`, as its model cards pass it): the token ids, every `stride`-th pixel
    or patch feature of the pixel values (and the patch grids and masks of SigLIP 2), the
    outputs of the model but its hidden states, and every `stride`-th token of those."""
    import io
    import urllib.request
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(repo, revision=sha, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(repo, revision=sha)
    with urllib.request.urlopen(COCO_URL) as f:
        image = Image.open(io.BytesIO(f.read())).convert("RGB").resize(size, Image.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous()
    padding = {} if type(processor).__name__ == "Siglip2Processor" else dict(padding="max_length")
    inputs = processor(images=[pixels], text=SIGLIP_TEXTS, return_tensors="pt", **padding)
    with torch.no_grad():
        outputs = model(**inputs)
    pixel_values = inputs["pixel_values"]
    pixel_values = (pixel_values[:, :, ::stride, ::stride] if pixel_values.ndim == 4
                    else pixel_values[:, ::stride // 2, ::stride])
    tensors = {"image": pixels, "input_ids": inputs["input_ids"], "pixel_values": pixel_values,
               **{k: inputs[k] for k in ("spatial_shapes", "pixel_attention_mask") if k in inputs}}
    for k in ("logits_per_image", "logits_per_text", "text_embeds", "image_embeds"):
        tensors[k] = getattr(outputs, k).clone()  # logits_per_image may be a view
    for part in ("text_model_output", "vision_model_output"):
        out = getattr(outputs, part)
        tensors[f"{part}/pooler_output"] = out.pooler_output
        tensors[f"{part}/last_hidden_state"] = out.last_hidden_state[:, ::stride]
    out = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "class": type(model).__name__,
            "texts": json.dumps(SIGLIP_TEXTS), "stride": str(stride), **versions()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out, metadata=meta)
    probs = outputs.logits_per_image.sigmoid()[0].tolist()
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB, probabilities "
          f"{[round(p, 4) for p in probs]}")


CLIP_TEXTS = ["a photo of a cat", "a photo of a dog", "a photo of a car",
              "two cats asleep on a pink couch, next to two remote controls"]


def dump_clip_reference(repo, sha, config, subfolder="", size=(160, 120), strides=(8, 8, 4)):
    """The outputs of a CLIP model in float32, of which large arrays are stored strided: every
    `strides[0]`-th pixel of the images, every `strides[1]`-th token and `strides[2]`-th
    feature of the hidden states.

    A `CLIPModel` takes `CLIP_TEXTS` and two images, that of `IMAGE_URL`, resized to `size`
    (w, h), and random noise of 48 × 64 pixels, stored as 8-bit images ("image/<i>",
    channels × height × width), through the repository's processor (`input_ids`,
    `attention_mask` and `pixel_values`, padded to the longest text). Stores the outputs of
    the model, `logits_per_image`, `logits_per_text`, `text_embeds` and `image_embeds`, and
    those of `get_text_features` and `get_image_features`: the projected and unnormalised
    features ("text/pooler_output", "image/pooler_output") and the hidden states
    ("text/last_hidden_state", "image/last_hidden_state").

    A `CLIPTextModel`, such as the text encoder of a Stable Diffusion pipeline (a
    `subfolder`), takes `CLIP_TEXTS` tokenized by the tokenizer of the pipeline (its
    `tokenizer` subfolder) as the pipeline does, padded to `max_length` with no attention
    mask, and stores its `pooler_output` and `last_hidden_state`."""
    pixels, tokens, features = strides
    import io
    import urllib.request
    from PIL import Image
    import transformers
    from transformers import AutoProcessor, CLIPTokenizer
    arch = config["architectures"][0]
    cls = getattr(transformers, arch)
    kwargs = dict(subfolder=subfolder) if subfolder else {}
    model = cls.from_pretrained(repo, revision=sha, dtype=torch.float32,
                                attn_implementation="sdpa", **kwargs).eval()
    name = f"{repo}/{subfolder}" if subfolder else repo
    tensors, meta = {}, {"repo": name, "revision": sha, "class": arch,
                         "texts": json.dumps(CLIP_TEXTS), "strides": json.dumps(strides),
                         **versions()}

    def hidden(x):
        return x[:, ::tokens, ::features]

    if arch == "CLIPTextModel":
        tokenizer = CLIPTokenizer.from_pretrained(repo, revision=sha, subfolder="tokenizer")
        ids = tokenizer(CLIP_TEXTS, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(input_ids=ids)
        tensors.update({"input_ids": ids, "pooler_output": out.pooler_output,
                        "last_hidden_state": hidden(out.last_hidden_state)})
    else:
        assert arch == "CLIPModel", arch
        with urllib.request.urlopen(IMAGE_URL) as f:
            photo = Image.open(io.BytesIO(f.read())).convert("RGB").resize(size, Image.BICUBIC)
        gen = torch.Generator().manual_seed(0)
        images = [torch.from_numpy(np.asarray(photo).copy()).permute(2, 0, 1),
                  torch.randint(0, 256, (3, 64, 48), generator=gen, dtype=torch.uint8)]
        processor = AutoProcessor.from_pretrained(repo, revision=sha)
        inputs = processor(text=CLIP_TEXTS, images=images, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            text = model.get_text_features(input_ids=inputs["input_ids"],
                                           attention_mask=inputs["attention_mask"])
            image = model.get_image_features(pixel_values=inputs["pixel_values"])
        tensors.update({f"image/{i + 1}": x for i, x in enumerate(images)})
        tensors.update({k: inputs[k] for k in ("input_ids", "attention_mask")})
        tensors["pixel_values"] = inputs["pixel_values"][:, :, ::pixels, ::pixels]
        tensors.update({k: getattr(out, k) for k in ("logits_per_image", "logits_per_text",
                                                     "text_embeds", "image_embeds")})
        tensors.update({"text/pooler_output": text.pooler_output,
                        "text/last_hidden_state": hidden(text.last_hidden_state),
                        "image/pooler_output": image.pooler_output,
                        "image/last_hidden_state": hidden(image.last_hidden_state)})
        probabilities = out.logits_per_image.softmax(dim=1)[0]
        print(f"{name}: the probabilities of the texts for the photo {probabilities.tolist()}")
    out_path = reference_path(name)
    save_file({k: v.contiguous() for k, v in tensors.items()}, out_path, metadata=meta)
    print(f"{out_path}: {name}@{sha}, {out_path.stat().st_size / 1e3:.1f} kB")
ENCODER_TEXTS = ["The capital of France is Paris.",
                 "A cat sat on the mat while the dog barked loudly at the passing cars.",
                 "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)"]
ENCODER_PAIRS = [("what is a panda?", "The giant panda is a bear native to central China."),
                 ("what is a panda?", "Paris is the capital and largest city of France."),
                 ("how many people live in Berlin?",
                  "Berlin had a population of 3,520,031 registered inhabitants in 2016.")]


def dump_encoder_reference(repo, sha):
    """The outputs of a text encoder (BertModel, XLMRobertaModel, ModernBertModel, or one with
    a head) on `ENCODER_TEXTS`, or, for a classifier, on `ENCODER_PAIRS` as (query, passage)
    pairs, which the repo's tokenizer pads as a batch: "input_ids" (0-based), "attention_mask"
    and "token_type_ids" if the tokenizer returns them; the "hidden_states" of the base model
    (all tokens, whose padding is not meaningful), its "pooler_output" if it has one, the
    "logits" of a classifier, and the "last_logits" of a masked language model, the logits of
    the last token of each text. The first text of a masked language model ends with a mask."""
    import transformers
    from transformers import AutoTokenizer
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    arch = config["architectures"][0]
    model = getattr(transformers, arch).from_pretrained(
        repo, revision=sha, dtype=torch.float32, attn_implementation="sdpa").eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha)
    classifier = arch.endswith("ForSequenceClassification")
    masked = arch.endswith("ForMaskedLM")
    texts = list(ENCODER_TEXTS)
    if masked:
        texts[0] = ENCODER_TEXTS[0].replace("Paris.", f"{tokenizer.mask_token}.")
    if classifier:
        queries, passages = zip(*ENCODER_PAIRS)
        enc = tokenizer(list(queries), list(passages), padding=True, return_tensors="pt")
    else:
        enc = tokenizer(texts, padding=True, return_tensors="pt")
    base = getattr(model, model.base_model_prefix, model)
    with torch.no_grad():
        out = model(**enc)
        base_out = base(**enc)
    tensors = {k: v for k, v in enc.items()}
    tensors["hidden_states"] = base_out.last_hidden_state
    if getattr(base_out, "pooler_output", None) is not None:
        tensors["pooler_output"] = base_out.pooler_output
    if classifier:
        tensors["logits"] = out.logits
    if masked:
        last = enc["attention_mask"].sum(1) - 1
        tensors["last_logits"] = out.logits[torch.arange(len(texts)), last]
    path = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "class": arch, **versions()}
    if classifier:
        meta["texts"] = json.dumps(list(queries))
        meta["text_pairs"] = json.dumps(list(passages))
    else:
        meta["texts"] = json.dumps(texts)
    save_reference(path, tensors, meta)
    shown = tensors["logits"].flatten().tolist() if classifier else tuple(tensors["hidden_states"].shape)
    print(f"{path}: {repo}@{sha}, {path.stat().st_size / 1e3:.1f} kB, {shown}")


def save_reference(path, tensors, meta):
    """Write a reference file, keeping what an earlier dump of the same revision wrote to it: a
    sentence-transformers repo has the embeddings of its model (`dump_sentence_transformers_
    reference`) and the outputs of its encoder (`dump_encoder_reference`) in one file, under
    different keys."""
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    if path.exists():
        with safe_open(path, "pt") as f:
            old = f.metadata()
        if old.get("revision") == meta["revision"]:
            tensors = {**load_file(path), **tensors}
            meta = {**old, **meta}
    save_file(tensors, path, metadata=meta)


def reference_path(name):
    out = ROOT / "test" / "references" / (name.replace("/", "--") + ".safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def half_randn(gen, *shape):
    """Standard normal values rounded to float16, returned in float32."""
    return torch.randn(*shape, generator=gen).half().float()


def dump_diffusers_reference(repo, subfolder, sha, batch):
    import diffusers
    config = json.loads(pathlib.Path(hf_hub_download(repo, f"{subfolder}/config.json", revision=sha)).read_text())
    cls = getattr(diffusers, config["_class_name"])
    model = cls.from_pretrained(repo, subfolder=subfolder, revision=sha, torch_dtype=torch.float32).eval()
    gen = torch.Generator().manual_seed(0)
    if config["_class_name"].startswith("Autoencoder"):
        shape = [batch, config["in_channels"], 64, 48]
        x = reference_input(shape)
        with torch.no_grad():
            latents = model.encode(x).latent_dist.mode()
            output = model.decode(latents).sample
        tensors = {"latents": latents, "output": output}
        meta = {"input_shape": json.dumps(shape)}
    elif config["_class_name"] == "Flux2Transformer2DModel":
        from diffusers import Flux2KleinPipeline as Pipeline
        h, w, n = 8, 6, 12
        hidden = half_randn(gen, batch, h * w, config["in_channels"])
        text = half_randn(gen, batch, n, config["joint_attention_dim"])
        timestep = torch.tensor([0.7, 0.3][:batch])
        img_ids = Pipeline._prepare_latent_ids(torch.zeros(batch, 1, h, w))
        txt_ids = Pipeline._prepare_text_ids(text)
        with torch.no_grad():
            output = model(hidden, text, timestep, img_ids, txt_ids).sample
        tensors = {"inputs/1": hidden.half(), "inputs/2": text.half(), "inputs/3": timestep,
                   "inputs/4": img_ids, "inputs/5": txt_ids, "output": output}
        meta = {}
    elif config["_class_name"] == "ZImageTransformer2DModel":
        # latents of 6 × 8 patches, padded to 64 tokens, and captions whose padded lengths
        # differ (32 and 64), which the attention masks
        x = half_randn(gen, batch, config["in_channels"], 1, 12, 16)
        caps = [half_randn(gen, n, config["cap_feat_dim"]) for n in [10, 40][:batch]]
        timestep = torch.tensor([0.7, 0.3][:batch])
        with torch.no_grad():
            output = torch.stack(model(list(x.unbind(0)), timestep, caps).sample)
        tensors = {"inputs/1": x.half(), "inputs/2": timestep, "output": output,
                   **{f"inputs/3/{j + 1}": c.half() for j, c in enumerate(caps)}}
        meta = {}
    else:
        raise SystemExit(f"no reference for {config['_class_name']}")
    name = f"{repo}/{subfolder}"
    out = reference_path(name)
    meta.update({"repo": name, "revision": sha, "class": config["_class_name"],
                 "num_params": str(sum(p.numel() for p in model.parameters())), **versions()})
    save_file({k: v.contiguous() for k, v in tensors.items()}, out, metadata=meta)
    print(f"{out}: {name}@{sha}, {out.stat().st_size / 1e3:.1f} kB")


PIPELINE_PROMPT = "A photo of a red fox sitting in fresh snow, looking at the camera"


EDIT_PROMPT = "Put a small red knitted hat on the cat"


def dump_pipeline_reference(repo, sha, height=256, width=256, steps=4, embed_positions=24,
                            edit_size=(200, 136)):
    """An image of a diffusers text-to-image pipeline, in float32, from fixed noise, and an
    edit of the image of `IMAGE_URL`, resized to `edit_size` (w, h), by `EDIT_PROMPT`.

    Stores the token ids and mask of the prompt (0-based), the prompt embeddings of its first
    `embed_positions` positions, the initial noise, the denoised latents and the image; and
    the same under "edit/..." for the edit (half as many positions of its embeddings), with
    its reference image ("edit/image", 8-bit), whose size the edit takes. The text encoder
    runs first, alone, to halve the memory."""
    import gc
    import io
    import urllib.request
    import diffusers
    from PIL import Image
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    Pipeline = diffusers.Flux2KleinPipeline
    index = json.loads(pathlib.Path(hf_hub_download(repo, "model_index.json", revision=sha)).read_text())
    if index["_class_name"] == "ZImagePipeline":
        return dump_z_image_reference(repo, sha, height, width, embed_positions=embed_positions)
    assert index["_class_name"] == "Flux2KleinPipeline", index["_class_name"]
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", revision=sha)
    text_encoder = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", revision=sha,
                                                    torch_dtype=torch.float32).eval()

    def tokens(prompt):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        return tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                         max_length=512)

    with torch.no_grad():  # as `pipe.encode_prompt(prompt)`
        embeds = {p: Pipeline._get_qwen3_prompt_embeds(text_encoder, tokenizer, p, device="cpu")
                  for p in (PIPELINE_PROMPT, EDIT_PROMPT)}
    del text_encoder
    gc.collect()

    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, revision=sha, text_encoder=None,
                                                       tokenizer=None, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    with urllib.request.urlopen(IMAGE_URL) as f:
        image = Image.open(io.BytesIO(f.read())).convert("RGB").resize(edit_size, Image.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    gen = torch.Generator().manual_seed(0)
    f = pipe.vae_scale_factor * 2
    channels = pipe.transformer.config.in_channels
    tensors = {}
    edit_h, edit_w = edit_size[1] // f * f, edit_size[0] // f * f  # the crop of the image
    for prefix, prompt, (h, w), positions, inputs in [
            ("", PIPELINE_PROMPT, (height, width), embed_positions,
             dict(height=height, width=width)),
            ("edit/", EDIT_PROMPT, (edit_h, edit_w), embed_positions // 2, dict(image=image))]:
        latents = half_randn(gen, 1, channels, h // f, w // f)
        kwargs = dict(prompt_embeds=embeds[prompt], num_inference_steps=steps, guidance_scale=1.0,
                      **inputs)
        with torch.no_grad():
            denoised = pipe(latents=latents.clone(), output_type="latent", **kwargs).images
            output = pipe(latents=latents.clone(), output_type="pt", **kwargs).images
        ids = tokens(prompt)
        tensors.update({prefix + "input_ids": ids["input_ids"],
                        prefix + "attention_mask": ids["attention_mask"],
                        prefix + "prompt_embeds": embeds[prompt][:, :positions],
                        prefix + "latents": latents.half(), prefix + "output_latents": denoised,
                        prefix + "output": output})
    tensors["edit/image"] = pixels
    out = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "class": index["_class_name"],
            "prompt": PIPELINE_PROMPT, "height": str(height), "width": str(width),
            "num_inference_steps": str(steps), "edit_prompt": EDIT_PROMPT,
            "edit_image": IMAGE_URL, **versions()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB")


def dump_z_image_reference(repo, sha, height, width, steps=9, guidance_scale=0.0,
                           embed_positions=24):
    """An image of Z-Image-Turbo with its settings (9 steps without guidance), as
    `dump_pipeline_reference` makes it. The text encoder and the rest of the pipeline load one
    after the other, to fit in less memory (16 and 25 GB in float32)."""
    import gc

    import diffusers
    load = dict(revision=sha, torch_dtype=torch.float32)
    pipe = diffusers.ZImagePipeline.from_pretrained(repo, transformer=None, vae=None, **load)
    text = pipe.tokenizer.apply_chat_template([{"role": "user", "content": PIPELINE_PROMPT}],
                                              tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True)
    tokens = pipe.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=512)
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(PIPELINE_PROMPT, do_classifier_free_guidance=False)
    del pipe
    gc.collect()
    pipe = diffusers.ZImagePipeline.from_pretrained(repo, text_encoder=None, tokenizer=None, **load)
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator().manual_seed(0)
    f = pipe.vae_scale_factor * 2
    latents = half_randn(gen, 1, pipe.transformer.config.in_channels, 2 * (height // f),
                         2 * (width // f))
    with torch.no_grad():
        kwargs = dict(prompt_embeds=prompt_embeds, height=height, width=width,
                      num_inference_steps=steps, guidance_scale=guidance_scale)
        denoised = pipe(latents=latents.clone(), output_type="latent", **kwargs).images
        image = pipe(latents=latents.clone(), output_type="pt", **kwargs).images
    out = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "class": type(pipe).__name__,
            "prompt": PIPELINE_PROMPT, "height": str(height), "width": str(width),
            "num_inference_steps": str(steps), "guidance_scale": str(guidance_scale),
            **versions()}
    tensors = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
               "prompt_embeds": prompt_embeds[0][:embed_positions], "latents": latents.half(),
               "output_latents": denoised, "output": image}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
