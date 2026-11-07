"""The fixture of `generate`: a tiny causal language model, saved with a generation_config.json,
and what transformers' `generate` gives for random prompts, in float64.

A transformers spec with "generate" has, there, the "lengths" of the prompts, the
"generation_config" of the checkpoint, to which `dump_generation` adds the end-of-text tokens
that the model generates (`eos_steps`: the token of sequence `b` at step `eos_steps[b]` of a
greedy run, 0-based, or the first after it that no sequence generates before its own step),
and "cases": keyword arguments of `generate`, `{}` for the settings of the checkpoint.

Each prompt is generated alone, so that the padding of a batch doesn't change the
repetition penalty (transformers penalizes the padding tokens of a batch padded on the left);
the outputs of a batch are those of its sequences, padded after the end-of-text token. The
fixture stores the prompts, padded on the right, as "input_ids" and "attention_mask", and for
case `i` and sequence `b` (from 1) the new tokens "cases/<i>/<b>/tokens"; when sampling, also
the logits of each step, "cases/<i>/<b>/logits" (`output_logits`), and the scores that the
logits processors make of them, "cases/<i>/<b>/scores" (`output_scores`).
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig

from common import build, exact_float64, rerandomize, versions


def dump_generation(spec, out, args):
    g = spec["generate"]
    seed = spec.get("seed", 0)
    torch.manual_seed(seed)
    model = build(spec).eval()
    rerandomize(model, seed)
    gen = torch.Generator().manual_seed(seed + 1)
    vocab_size = model.config.get_text_config().vocab_size
    prompts = [torch.randint(vocab_size, (1, n), generator=gen) for n in g["lengths"]]

    double = build(spec).eval()
    double.load_state_dict(model.state_dict())
    double = double.double()
    exact_float64(double)

    def generate(ids, settings, **kwargs):
        double.generation_config = settings
        with torch.no_grad():
            out = double.generate(ids, attention_mask=torch.ones_like(ids), **kwargs,
                                  return_dict_in_generate=True, output_scores=True,
                                  output_logits=True)
        return out

    # the end-of-text tokens: tokens of a greedy run without them, at a step of a sequence or
    # the first after it that no sequence generates before its own step
    settings = GenerationConfig(**g["generation_config"])
    runs = [generate(p, settings).sequences[0, p.shape[1]:].tolist() for p in prompts]
    steps = g["eos_steps"]
    early = {t for run, step in zip(runs, steps) for t in run[:step]}
    eos = []
    for run, step in zip(runs, steps):
        eos.append(next(t for t in run[step:] if t not in early and t not in eos))
    settings = GenerationConfig(**g["generation_config"], eos_token_id=eos)
    model.generation_config = settings
    model.save_pretrained(out)

    tensors = {}
    n = max(g["lengths"])
    input_ids = torch.zeros(len(prompts), n, dtype=torch.long)
    attention_mask = torch.zeros(len(prompts), n, dtype=torch.long)
    for b, p in enumerate(prompts):
        input_ids[b, :p.shape[1]] = p[0]
        attention_mask[b, :p.shape[1]] = 1
    tensors["input_ids"], tensors["attention_mask"] = input_ids, attention_mask
    summary = []
    for i, case in enumerate(g["cases"], 1):
        lengths = []
        for b, p in enumerate(prompts, 1):
            torch.manual_seed(seed + 100 * i + b)
            result = generate(p, settings, **case)
            new = result.sequences[0, p.shape[1]:]
            tensors[f"cases/{i}/{b}/tokens"] = new.contiguous()
            if case.get("do_sample", settings.do_sample):
                tensors[f"cases/{i}/{b}/logits"] = torch.cat(result.logits).contiguous()
                tensors[f"cases/{i}/{b}/scores"] = torch.cat(result.scores).contiguous()
            lengths.append(len(new))
        summary.append(f"{json.dumps(case)}: {lengths} new tokens")

    meta = {"spec": json.dumps(spec), "eos_token_id": json.dumps(eos), **versions()}
    save_file(tensors, out / "acts.safetensors", metadata=meta)
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: end-of-text tokens {eos}, {size / 1e6:.2f} MB")
    for line in summary:
        print("  ", line)
    return size <= args.max_bytes or bool(args.out)
