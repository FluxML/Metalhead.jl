"""Write a tiny parity fixture for a Metalhead port.

    uv run dump.py specs/<name>.json [specs/<other>.json ...] [--all-taps]
    uv run dump.py specs/<name>.json --out DIR --all-taps     # a debug dump

A spec with "library": "transformers" describes a transformers model: "architecture" (the class,
e.g. "LlamaForCausalLM"), "model_type" and "config" (the fields of its config), and "input_shape"
(batch, seq_len) of random token ids. Its fixture directory is written by `save_pretrained`, and
the RMSNorms and rotary tables are computed in float64 (see `exact_float64` in common.py).
With "lengths", the batch is padded on the right and the model takes an `attention_mask`.
A vision encoder (Qwen3_5VisionModel) has "grid_thw" instead of "input_shape": the
(t, h, w) grids of patches of its images (see `dump_transformers_vision`).

A spec with "library": "diffusers" describes a diffusers model, "architecture" (the class, e.g.
"Flux2Transformer2DModel") with the arguments "config", saved with `save_pretrained`, and the
sizes of its inputs (see `diffusers_inputs`). A "Flux2KleinPipeline" spec has "components"
instead, and is run end to end (see `dump_flux2_klein`).

Optional spec keys: "shrink": k divides every channel count by k (see `shrunk_channels` in
common.py; the Julia side applies `shrink_channels` from config.json), and
"weight_dtype": "float16" stores the weights in half precision, and "tap_depth": d records the
modules with at most d dots in their name (default 1), to keep large fixtures small.

The fixture directory looks like a Hub repository (config.json + model.safetensors), so the Julia
side loads it with `create_model(dir; pretrained = true)`, plus acts.safetensors with the input,
logits, features, module outputs ("taps") and `forward_intermediates` of the PyTorch model,
computed in float64.
"""
import argparse
import json
import pathlib

import torch
from safetensors.torch import save_file

from common import (build, build_transformers, classifier_name, exact_float64,
                    exact_float64_diffusers, float64_pos_embed, rerandomize, state_dict_f32,
                    versions, write_json)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+", metavar="spec")
    p.add_argument("--out", help="output directory, for a single spec (default: test/fixtures/<spec name>)")
    p.add_argument("--all-taps", action="store_true", help="record every module, not only depth <= 2")
    p.add_argument("--max-bytes", type=int, default=1_000_000)
    p.add_argument("--tap-dtype", choices=["float16", "float32"],
                   help="storage type of the taps (default: float16, or float32 with --all-taps)")
    args = p.parse_args()
    if args.out and len(args.specs) > 1:
        p.error("--out takes a single spec")
    too_large = [s for s in args.specs if not dump(pathlib.Path(s), args)]
    if too_large:
        raise SystemExit(f"fixtures larger than {args.max_bytes} bytes, shrink their specs: {too_large}")


def dump(spec_path, args):
    """Write the fixture of one spec; return whether it fits in `args.max_bytes`."""
    spec = json.loads(spec_path.read_text())
    out = pathlib.Path(args.out) if args.out else ROOT / "test" / "fixtures" / spec_path.stem
    out.mkdir(parents=True, exist_ok=True)

    if spec.get("library") == "transformers":
        return dump_transformers(spec, out, args)
    if spec.get("library") == "diffusers":
        return dump_diffusers(spec, out, args)

    torch.manual_seed(spec.get("seed", 0))
    model = build(spec).eval()
    rerandomize(model, spec.get("seed", 0))
    weights = state_dict_f32(model)
    if spec.get("weight_dtype") == "float16":  # halves the file; torch gets the rounded values
        weights = {k: v.half() if v.is_floating_point() else v for k, v in weights.items()}
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in weights.items()})
    model = model.double()

    tap_dtype = getattr(torch, args.tap_dtype or ("float32" if args.all_taps else "float16"))
    taps, order, counts = {}, [], {}

    def hook(name):
        def fn(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            counts[name] = counts.get(name, 0) + 1
            for i, o in enumerate(outs):
                if isinstance(o, torch.Tensor):
                    key = f"{name}#{counts[name]}" + (f".{i + 1}" if len(outs) > 1 else "")
                    taps["tap/" + key] = o.detach().to(tap_dtype).contiguous()
                    order.append(key)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if n and (args.all_taps or n.count(".") <= spec.get("tap_depth", 1))]
    gen = torch.Generator().manual_seed(spec.get("seed", 0) + 1)
    x = torch.randn(spec["input_shape"], generator=gen, dtype=torch.float64)
    with torch.no_grad(), float64_pos_embed():
        logits = model(x)
        for h in handles:
            h.remove()
        features = model.forward_features(x)
        intermediates = dump_intermediates(model, x)

    kwargs = dict(spec.get("kwargs", {}))
    num_classes = kwargs.pop("num_classes", getattr(model, "num_classes", None))
    config = {
        "architecture": spec["architecture"],
        "num_classes": num_classes,
        "model_args": kwargs,
        "pretrained_cfg": {"input_size": spec["input_shape"][1:], "classifier": classifier_name(model)},
    }
    if spec.get("shrink"):
        config["shrink"] = spec["shrink"]
    write_json(out / "config.json", config)
    save_file(weights, out / "model.safetensors")
    acts = {"input": x, "output": logits.contiguous(), "features": features.contiguous(), **taps,
            **intermediates}
    meta = {"spec": json.dumps(spec), "call_order": json.dumps(order),
            "num_params": str(sum(p.numel() for p in model.parameters())), **versions()}
    if intermediates:
        info = model.feature_info
        info = info.get_dicts() if hasattr(info, "get_dicts") else info
        table = []
        for i, f in enumerate(info):
            channels = f["num_chs"]
            actual = intermediates[f"intermediates/{i + 1}"].shape[1]
            if channels != actual:
                assert spec.get("shrink") and channels == actual * spec["shrink"], (f, actual)
                channels = actual
            table.append(dict(num_chs=channels, reduction=f["reduction"], module_name=f["module"]))
        meta["feature_info"] = json.dumps(table)
    save_file(acts, out / "acts.safetensors", metadata=meta)

    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: {len(weights)} tensors, {len(taps)} taps, {len(intermediates)} intermediates, "
          f"{size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)


def dump_intermediates(model, x):
    """The native intermediates, or feature_info hook outputs, in float64.

    "intermediates/<i>" (1-based) are the NCHW intermediates, "intermediates_norm/<i>" those with
    `norm=True` when they differ, and "intermediates_prefix/<i>" the prefix (class and register)
    tokens of vision transformers.
    """
    store = lambda y: y.detach().clone().contiguous()
    if not hasattr(model, "forward_intermediates"):
        info = getattr(model, "feature_info", [])
        if not info:
            return {}
        taps = {}
        def hook(i):
            def fn(module, inputs, output):
                taps[f"intermediates/{i + 1}"] = store(output)
            return fn
        handles = [model.get_submodule(f["module"]).register_forward_hook(hook(i))
                   for i, f in enumerate(info)]
        try:
            model.forward_features(x)
        finally:
            for handle in handles:
                handle.remove()
        return taps
    _, plain = model.forward_intermediates(x)
    out = {f"intermediates/{i + 1}": store(y) for i, y in enumerate(plain)}
    _, normed = model.forward_intermediates(x, norm=True)
    if any(not torch.equal(a, b) for a, b in zip(plain, normed)):
        out.update({f"intermediates_norm/{i + 1}": store(y) for i, y in enumerate(normed)})
    if getattr(model, "num_prefix_tokens", 0) > 0:
        _, pairs = model.forward_intermediates(x, return_prefix_tokens=True)
        out.update({f"intermediates_prefix/{i + 1}": store(p) for i, (_, p) in enumerate(pairs)})
    return out


def dump_transformers(spec, out, args):
    """Write the fixture of a transformers spec: the logits of a causal LM, its last hidden state
    (features), the outputs of its decoder layers (intermediates, plain and with the final norm)
    and those of the modules with parameters (taps).

    With "lengths" (one per sequence of the batch), the sequences are padded on the right to
    the length of `input_shape`, as by a tokenizer, and the model takes their `attention_mask`,
    stored as "kwargs/attention_mask"."""
    if "grid_thw" in spec:
        return dump_transformers_vision(spec, out, args)
    seed = spec.get("seed", 0)
    torch.manual_seed(seed)
    model = build(spec).eval()
    rerandomize(model, seed)
    model.save_pretrained(out)  # config.json, and model.safetensors without the tied lm_head
    (out / "generation_config.json").unlink(missing_ok=True)
    model = model.double()
    exact_float64(model)

    tap_dtype = getattr(torch, args.tap_dtype or ("float32" if args.all_taps else "float16"))
    taps, order, counts = {}, [], {}

    def hook(name):
        def fn(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            tensors = [o for o in outs if isinstance(o, torch.Tensor)]
            counts[name] = counts.get(name, 0) + 1
            for i, o in enumerate(tensors):
                key = f"{name}#{counts[name]}" + (f".{i + 1}" if len(tensors) > 1 else "")
                taps["tap/" + key] = o.detach().to(tap_dtype).contiguous()
                order.append(key)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if n and any(True for _ in m.parameters())
               and (args.all_taps or n.count(".") <= spec.get("tap_depth", 1))]
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randint(model.config.vocab_size, spec["input_shape"], generator=gen)
    kwargs = {}
    if "lengths" in spec:
        positions = torch.arange(x.shape[1])
        kwargs["attention_mask"] = torch.stack([positions < n for n in spec["lengths"]]).long()
    with torch.no_grad():
        logits = model(input_ids=x, **kwargs).logits
        for h in handles:
            h.remove()
        features = model.model(input_ids=x, **kwargs).last_hidden_state
        layers = []
        handles = [layer.register_forward_hook(lambda m, i, o: layers.append(o.detach().clone()))
                   for layer in model.model.layers]
        model.model(input_ids=x, **kwargs)
        for h in handles:
            h.remove()
        intermediates = {f"intermediates/{i + 1}": y.contiguous() for i, y in enumerate(layers)}
        intermediates.update({f"intermediates_norm/{i + 1}": model.model.norm(y).contiguous()
                              for i, y in enumerate(layers)})

    acts = {"input": x, "output": logits.contiguous(), "features": features.contiguous(), **taps,
            **intermediates, **{f"kwargs/{k}": v for k, v in kwargs.items()}}
    meta = {"spec": json.dumps(spec), "call_order": json.dumps(order),
            "num_params": str(sum(p.numel() for p in model.parameters())), **versions()}
    save_file(acts, out / "acts.safetensors", metadata=meta)

    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: {len(model.state_dict())} tensors, {len(taps)} taps, {size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)


def dump_transformers_vision(spec, out, args):
    """Write the fixture of a transformers vision encoder (Qwen3_5VisionModel) of images and
    videos, which `spec["grid_thw"]` sizes: a list of [t, h, w] grids of patches. Its inputs
    are random patches, laid out as the image processor does ("inputs/1", (num_patches,
    patch_dim)), and the grids ("inputs/2"); its outputs "output/last_hidden_state" and
    "output/pooler_output" (the merged tokens)."""
    seed = spec.get("seed", 0)
    torch.manual_seed(seed)
    model = build(spec).eval()
    rerandomize(model, seed)
    model.save_pretrained(out)
    model = model.double()
    exact_float64(model)

    tap_dtype = getattr(torch, args.tap_dtype or ("float32" if args.all_taps else "float16"))
    taps, order = {}, []
    names = {n for n, m in model.named_modules()
             if n and any(True for _ in m.parameters())
             and (args.all_taps or n.count(".") <= spec.get("tap_depth", 1))}
    cfg = model.config
    gen = torch.Generator().manual_seed(seed + 1)
    grid_thw = torch.tensor(spec["grid_thw"])
    patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
    pixel_values = torch.randn(int(grid_thw.prod(-1).sum()), patch_dim, generator=gen,
                               dtype=torch.float64)
    with torch.no_grad():
        handles = tap_hooks(model, names, taps, order, tap_dtype)
        outputs = model(pixel_values, grid_thw=grid_thw)
        for h in handles:
            h.remove()

    acts = {"inputs/1": pixel_values, "inputs/2": grid_thw,
            "output/last_hidden_state": outputs.last_hidden_state.contiguous(),
            "output/pooler_output": outputs.pooler_output.contiguous(), **taps}
    meta = {"spec": json.dumps(spec), "call_order": json.dumps(order),
            "num_params": str(sum(p.numel() for p in model.parameters())), **versions()}
    save_file(acts, out / "acts.safetensors", metadata=meta)

    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: {len(model.state_dict())} tensors, {len(taps)} taps, {size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)


def diffusers_inputs(spec, model, gen):
    """The positional inputs of the forward pass of a diffusers model, as `spec["inputs"]` sizes
    them."""
    sizes = spec["inputs"]
    if spec["architecture"] == "Flux2Transformer2DModel":
        from diffusers import Flux2KleinPipeline as Pipeline
        b, h, w, n = sizes["batch"], sizes["height"], sizes["width"], sizes["text_len"]
        cfg = model.config
        hidden = torch.randn(b, h * w, cfg.in_channels, generator=gen, dtype=torch.float64)
        text = torch.randn(b, n, cfg.joint_attention_dim, generator=gen, dtype=torch.float64)
        timestep = torch.rand(b, generator=gen, dtype=torch.float64)
        img_ids = Pipeline._prepare_latent_ids(torch.zeros(b, 1, h, w))
        txt_ids = Pipeline._prepare_text_ids(text)
        inputs = [hidden, text, timestep, img_ids, txt_ids]
        if cfg.guidance_embeds:
            inputs.append(1 + 4 * torch.rand(b, generator=gen, dtype=torch.float64))
        return inputs
    return [torch.randn(sizes["sample"], generator=gen, dtype=torch.float64)]


def tap_hooks(model, names, taps, order, tap_dtype):
    """Record the tensor outputs of the modules `names` of `model` as taps, in call order."""
    counts = {}

    def hook(name):
        def fn(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            tensors = [o for o in outs if isinstance(o, torch.Tensor)]
            counts[name] = counts.get(name, 0) + 1
            for i, o in enumerate(tensors):
                key = f"{name}#{counts[name]}" + (f".{i + 1}" if len(tensors) > 1 else "")
                taps["tap/" + key] = o.detach().to(tap_dtype).contiguous()
                order.append(key)
        return fn

    return [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n in names]


def dump_diffusers(spec, out, args):
    """Write the fixture of a diffusers spec: the output of the model on random inputs
    ("inputs/<i>"), and the outputs of its modules (taps)."""
    if spec["architecture"] == "Flux2KleinPipeline":
        return dump_flux2_klein(spec, out, args)
    seed = spec.get("seed", 0)
    torch.manual_seed(seed)
    model = build(spec).eval()
    rerandomize(model, seed)
    model.save_pretrained(out)  # config.json and diffusion_pytorch_model.safetensors
    model = model.double()

    tap_dtype = getattr(torch, args.tap_dtype or ("float32" if args.all_taps else "float16"))
    taps, order = {}, []
    names = {n for n, m in model.named_modules()
             if n and (args.all_taps or n.count(".") <= spec.get("tap_depth", 1))}
    gen = torch.Generator().manual_seed(seed + 1)
    inputs = diffusers_inputs(spec, model, gen)
    with torch.no_grad(), exact_float64_diffusers():
        handles = tap_hooks(model, names, taps, order, tap_dtype)
        output = model(*inputs).sample
        for h in handles:
            h.remove()

    acts = {"output": output.contiguous(), **taps,
            **{f"inputs/{i + 1}": x.contiguous() for i, x in enumerate(inputs)}}
    meta = {"spec": json.dumps(spec), "call_order": json.dumps(order),
            "num_params": str(sum(p.numel() for p in model.parameters())), **versions()}
    save_file(acts, out / "acts.safetensors", metadata=meta)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {len(model.state_dict())} tensors, {len(taps)} taps, {size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)


def padded(gen, vocab_size, lengths, seq_len):
    """Random token ids, padded on the right to `seq_len` after `lengths`, and their mask."""
    ids = torch.randint(vocab_size, (len(lengths), seq_len), generator=gen)
    mask = torch.stack([torch.arange(seq_len) < n for n in lengths]).long()
    return ids, mask


def dump_flux2_klein(spec, out, args):
    """Write the fixture of a Flux2KleinPipeline of tiny components: the directory of the
    pipeline (model_index.json and a subfolder for each component, without a tokenizer), and
    the prompt embeddings of random token ids, the initial noise, the denoised latents and the
    images of the pipeline, in float64.

    Spec keys: "components" ("vae", "transformer", "text_encoder" as transformers specs,
    "scheduler"), "is_distilled", "lengths" and "seq_len" of the prompts ("negative_lengths" for
    classifier-free guidance), "height", "width", "num_inference_steps", "guidance_scale" and
    "text_encoder_out_layers"."""
    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           Flux2KleinPipeline, Flux2Transformer2DModel)
    seed = spec.get("seed", 0)
    torch.manual_seed(seed)
    parts = spec["components"]
    vae = AutoencoderKLFlux2.from_config(parts["vae"]).eval()
    transformer = Flux2Transformer2DModel.from_config(parts["transformer"]).eval()
    text_encoder = build_transformers(parts["text_encoder"]).eval()
    for i, m in enumerate((vae, transformer, text_encoder)):
        rerandomize(m, seed + i)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(parts["scheduler"])
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                              tokenizer=None, transformer=transformer,
                              is_distilled=spec.get("is_distilled", False))
    pipe.save_pretrained(out)
    (out / "text_encoder" / "generation_config.json").unlink(missing_ok=True)
    for m in (vae, transformer, text_encoder):
        m.double()
    exact_float64(text_encoder)

    layers = spec["text_encoder_out_layers"]
    gen = torch.Generator().manual_seed(seed + 1)
    vocab = text_encoder.config.vocab_size

    def embed(ids, mask):  # `_get_qwen3_prompt_embeds` after tokenization
        hidden = text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                              use_cache=False).hidden_states
        x = torch.stack([hidden[k] for k in layers], dim=1)
        b, c, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, c * d)

    tensors = {}
    with torch.no_grad(), exact_float64_diffusers():
        ids, mask = padded(gen, vocab, spec["lengths"], spec["seq_len"])
        prompt_embeds = embed(ids, mask)
        tensors.update({"input_ids": ids, "attention_mask": mask, "prompt_embeds": prompt_embeds})
        negative = None
        if "negative_lengths" in spec:
            neg_ids, neg_mask = padded(gen, vocab, spec["negative_lengths"], spec["seq_len"])
            negative = embed(neg_ids, neg_mask)
            tensors.update({"negative_input_ids": neg_ids, "negative_attention_mask": neg_mask,
                            "negative_prompt_embeds": negative})
        f = pipe.vae_scale_factor * 2
        shape = (ids.shape[0], transformer.config.in_channels, spec["height"] // f, spec["width"] // f)
        latents = torch.randn(shape, generator=gen, dtype=torch.float64)
        kwargs = dict(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative,
                      height=spec["height"], width=spec["width"],
                      num_inference_steps=spec["num_inference_steps"],
                      guidance_scale=spec.get("guidance_scale", 4.0))
        denoised = pipe(latents=latents.clone(), output_type="latent", **kwargs).images
        image = pipe(latents=latents.clone(), output_type="pt", **kwargs).images
    tensors.update({"latents": latents, "output_latents": denoised.contiguous(),
                    "output": image.contiguous()})
    meta = {"spec": json.dumps(spec), **versions()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out / "acts.safetensors", metadata=meta)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: pipeline, {size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)


if __name__ == "__main__":
    main()
