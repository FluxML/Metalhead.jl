"""Write reference outputs of a real pretrained checkpoint.

    uv run dump_reference.py timm/<model> [<org>/<model> ...] [--revision SHA]

Writes test/references/<org>--<model>.safetensors for each repo, with the logits and
pre-logits features of the checkpoint on `reference_input`, computed in float32. The metadata records the repo and
the commit sha, which the Julia test pins.

For a transformers causal LM, the input is `PROMPTS` tokenized by the repo's tokenizer and cut to
a common length (`input_ids`, 0-based), and the outputs are the hidden states after the final
norm, the logits of the last position, and the greedy continuation of the first prompt
(`generated_ids`), in float32, so that the Julia tests need neither Python nor a tokenizer.

A diffusers component is named `<org>/<repo>/<subfolder>` (e.g. `.../vae`): an autoencoder
encodes and decodes `reference_input` images, and a FLUX.2 or Z-Image transformer predicts the
flow of random latents; the metadata has its parameter count. A diffusers pipeline
(`<org>/<repo>` with a model_index.json) generates an image from `PIPELINE_PROMPT` end to end,
from fixed initial noise. Random inputs are stored rounded to float16, so that both sides use
the same values.
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
    args = p.parse_args()
    for repo in args.repos:
        dump_reference(repo, args.revision, args.batch)


PROMPTS = ["The capital of France is Paris, and the capital of Italy is",
           "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]


def dump_reference(repo, revision, batch):
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
    if "model_type" in config:
        return dump_lm_reference(repo, sha)
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


def dump_lm_reference(repo, sha, max_new_tokens=8):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    config = json.loads(pathlib.Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
    # a vision-language model: AutoModelForCausalLM would build its text model only
    auto = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    model = auto.from_pretrained(repo, revision=sha, dtype=torch.float32,
                                 attn_implementation="sdpa",
                                 experts_implementation="eager").eval()
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
    tensors = {"input_ids": input_ids, "hidden_states": hidden.contiguous(),
               "last_logits": logits.contiguous(), "generated_ids": generated.contiguous()}
    if "vision_config" in config:
        tensors.update(image_reference(model, repo, sha, max_new_tokens))
        meta["image_prompt"] = IMAGE_PROMPT
    save_file(tensors, out, metadata=meta)
    print(f"{out}: {repo}@{sha}, {out.stat().st_size / 1e3:.1f} kB, greedy continuation "
          f"{tokenizer.decode(generated[0, n:])!r}")


IMAGE_PROMPT = "Describe this image in one sentence."
IMAGE_URL = ("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
             "pipeline-cat-chonk.jpeg")


def image_reference(model, repo, sha, max_new_tokens, size=(224, 160)):
    """The outputs of a vision-language model on `IMAGE_PROMPT` about the image of `IMAGE_URL`,
    resized to `size` (w, h) and preprocessed by the repo's processor with `min_pixels` low
    enough to keep its size, under keys "image/...". The pixel values are rounded to float16."""
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
    return {"image/input_ids": inputs["input_ids"],
            "image/pixel_values": inputs["pixel_values"].half(),
            "image/image_grid_thw": inputs["image_grid_thw"],
            "image/hidden_states": hidden.contiguous(), "image/last_logits": logits.contiguous(),
            "image/generated_ids": generated.contiguous()}


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


def dump_pipeline_reference(repo, sha, height=256, width=256, steps=4, embed_positions=24):
    """An image of a diffusers text-to-image pipeline, in float32, from fixed noise.

    Stores the token ids and mask of the prompt (0-based), the prompt embeddings of its first
    `embed_positions` positions, the initial noise, the denoised latents and the image."""
    import diffusers
    index = json.loads(pathlib.Path(hf_hub_download(repo, "model_index.json", revision=sha)).read_text())
    if index["_class_name"] == "ZImagePipeline":
        return dump_z_image_reference(repo, sha, height, width, embed_positions=embed_positions)
    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, revision=sha, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    assert type(pipe).__name__ == "Flux2KleinPipeline", type(pipe).__name__
    tokenizer = pipe.tokenizer
    text = tokenizer.apply_chat_template([{"role": "user", "content": PIPELINE_PROMPT}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    gen = torch.Generator().manual_seed(0)
    f = pipe.vae_scale_factor * 2
    latents = half_randn(gen, 1, pipe.transformer.config.in_channels, height // f, width // f)
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(PIPELINE_PROMPT)
        kwargs = dict(prompt_embeds=prompt_embeds, height=height, width=width,
                      num_inference_steps=steps, guidance_scale=1.0)
        denoised = pipe(latents=latents.clone(), output_type="latent", **kwargs).images
        image = pipe(latents=latents.clone(), output_type="pt", **kwargs).images
    out = reference_path(repo)
    meta = {"repo": repo, "revision": sha, "class": type(pipe).__name__,
            "prompt": PIPELINE_PROMPT, "height": str(height), "width": str(width),
            "num_inference_steps": str(steps), **versions()}
    tensors = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
               "prompt_embeds": prompt_embeds[:, :embed_positions], "latents": latents.half(),
               "output_latents": denoised, "output": image}
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
