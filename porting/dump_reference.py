"""Write reference outputs of a real pretrained checkpoint.

    uv run dump_reference.py timm/<model> [<org>/<model> ...] [--revision SHA]

Writes test/references/<org>--<model>.safetensors for each repo, with the logits and
pre-logits features of the checkpoint on `reference_input`, computed in float32. The metadata records the repo and
the commit sha, which the Julia test pins.

For a transformers causal LM, the input is `PROMPTS` tokenized by the repo's tokenizer and cut to
a common length (`input_ids`, 0-based), and the outputs are the hidden states after the final
norm, the logits of the last position, and the greedy continuation of the first prompt
(`generated_ids`), in float32, so that the Julia tests need neither Python nor a tokenizer.
With `--layers k`, the model keeps the first k decoder layers of the checkpoint (and its
embeddings, final norm, head and vision encoder), for a checkpoint too large to run whole in
float32; the metadata records `num_hidden_layers`.

A diffusers component is named `<org>/<repo>/<subfolder>` (e.g. `.../vae`): an autoencoder
encodes and decodes `reference_input` images, and a FLUX.2 or Z-Image transformer predicts the
flow of random latents; the metadata has its parameter count. A diffusers pipeline
(`<org>/<repo>` with a model_index.json) generates an image from `PIPELINE_PROMPT` end to end,
from fixed initial noise. FLUX.2 [klein] also edits the image of `IMAGE_URL` as `EDIT_PROMPT`
says. Random inputs are stored rounded to float16, so that both sides use the same values.
"""
import argparse
import json

import numpy as np
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
    p.add_argument("--layers", type=int, default=None,
                   help="keep the first LAYERS decoder layers of a language model")
    args = p.parse_args()
    for repo in args.repos:
        dump_reference(repo, args.revision, args.batch, args.layers)


PROMPTS = ["The capital of France is Paris, and the capital of Italy is",
           "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]


def dump_reference(repo, revision, batch, layers=None):
    parts = repo.split("/")
    repo, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    sha = HfApi().model_info(repo, revision=revision).sha
    if subfolder:
        return dump_diffusers_reference(repo, subfolder, sha, batch)
    if any(f.rfilename == "model_index.json" for f in HfApi().model_info(repo, revision=sha).siblings):
        return dump_pipeline_reference(repo, sha)
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    if config.get("model_type") in ("sam3", "sam3_video"):
        return dump_sam3_reference(repo, sha)
    if config.get("model_type") in ("siglip", "siglip2"):
        return dump_siglip_reference(repo, sha)
    if "model_type" in config:
        return dump_lm_reference(repo, sha, layers=layers)
    if layers is not None:
        raise ValueError("--layers needs a transformers language model")
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


def dump_lm_reference(repo, sha, max_new_tokens=8, layers=None):
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoTokenizer)
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    # a vision-language model: AutoModelForCausalLM would build its text model only
    auto = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    kwargs = {}
    if layers is not None:
        # the weights of the other layers are left unused
        kwargs["config"] = truncated_config(AutoConfig.from_pretrained(repo, revision=sha), layers)
    model = auto.from_pretrained(repo, revision=sha, dtype=torch.float32,
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
    """`config` with the first `layers` decoder layers (of its `text_config`, if it has one)."""
    text = getattr(config, "text_config", config)
    if not 0 < layers <= text.num_hidden_layers:
        raise ValueError(f"--layers must be in 1:{text.num_hidden_layers}, got {layers}")
    if getattr(text, "layer_types", None) is not None:
        text.layer_types = text.layer_types[:layers]
    text.num_hidden_layers = layers
    return config


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
