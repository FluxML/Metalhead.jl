"""Fixtures of sentence-transformers models, written by dump.py for specs with
"library": "sentence-transformers".

A model spec builds a SentenceTransformer of a tiny transformers model ("transformer", a
transformers spec without "library") and a tokenizer of numbered words ("tokenizer": the
number of words "vocab", from 0, the "template" of its post-processor, a "lowercase"
normalizer, its "model_max_length" and "padding_side"), followed by the "modules" Pooling, Dense and Normalize
(each a dict of the arguments of its class, with its "type"). "sentence_bert_config" holds the
arguments of the Transformer module, "model" those of the SentenceTransformer (prompts, ...).
It is saved with sentence-transformers' `save`, in the format of sentence-transformers < 5 with
"legacy_configs" (pooling modes as booleans, modules.json types under
`sentence_transformers.models`, a Normalize without config.json, as in older Hub repositories),
and with "checkpoint": "head" the transformers model is saved with its head (keys under
`model.`, as Qwen3-Embedding's config.json names `Qwen3ForCausalLM`), which AutoModel strips.

The saved model is loaded again and run in float64 on the "texts": for each of the "calls",
the outputs of `encode` (or "method": "encode_query", "encode_document") with its arguments in
batches of "batch_size" ("calls/<i>/embeddings"), and the inputs that `preprocess` makes of all
the texts with the prompt that the call resolves ("calls/<i>/input_ids", 0-based,
"attention_mask", "prompt_length") with the outputs of the model on them
("token_embeddings", "sentence_embedding").

A "pooling" spec checks the Pooling module alone, on random token embeddings of
`batch × seq_len × dim` and the masks of "lengths", padded on the right and on the left
("right/attention_mask", "left/attention_mask"): the output of each pooling mode, and of all
of them concatenated, with the prompt included or not ("<side>/<mode>/<include_prompt>").
"""
import json
import pathlib
import shutil
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn

from common import build_transformers, exact_float64, rerandomize, versions, write_json

SPECIALS = ["<|pad|>", "<|unk|>", "<|cls|>", "<|sep|>", "<|eos|>"]
POOLING_MODES = ["cls", "max", "mean", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"]


def dump_sentence_transformers(spec, out, args):
    if "pooling" in spec:
        return dump_pooling(spec, out, args)
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules

    seed = spec.get("seed", 0)
    torch.manual_seed(seed)
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = pathlib.Path(tmp)
        transformer = build_transformers(spec["transformer"]).eval()
        rerandomize(transformer, seed)
        transformer.save_pretrained(model_dir)
        (model_dir / "generation_config.json").unlink(missing_ok=True)
        tokenizer(spec["tokenizer"]).save_pretrained(model_dir)

        # legacy configurations set them in sentence_bert_config.json instead
        kwargs = {} if spec.get("legacy_configs") else spec.get("sentence_bert_config", {})
        st_modules = [modules.Transformer(str(model_dir), **kwargs,
                                          model_kwargs={"attn_implementation": "sdpa"})]
        dim = transformer.config.hidden_size
        for i, m in enumerate(spec["modules"]):
            kwargs = {k: v for k, v in m.items() if k != "type"}
            if m["type"] == "Pooling":
                module = modules.Pooling(dim, **kwargs)
                dim = module.get_embedding_dimension()
            elif m["type"] == "Dense":
                act = kwargs.pop("activation_function", "Tanh")
                module = modules.Dense(dim, activation_function=getattr(nn, act)(), **kwargs)
                rerandomize(module, seed + i + 1)
                dim = module.out_features
            elif m["type"] == "Normalize":
                module = modules.Normalize(**kwargs)
            else:
                raise SystemExit(f"no fixture for a {m['type']} module")
            st_modules.append(module)
        SentenceTransformer(modules=st_modules, device="cpu", **spec.get("model", {})).save(
            str(out), create_model_card=False)
        if spec.get("checkpoint") == "head":
            for f in ("config.json", "model.safetensors"):
                shutil.copy(model_dir / f, out / f)
    if spec.get("legacy_configs"):
        legacy_configs(out, spec.get("sentence_bert_config", {}))

    st = SentenceTransformer(str(out), device="cpu",
                             model_kwargs={"attn_implementation": "sdpa"}).double()
    exact_float64(st[0].auto_model)
    texts = spec["texts"]
    tensors = {}
    with torch.no_grad():
        for i, call in enumerate(spec["calls"]):
            kwargs = {k: v for k, v in call.items() if k != "method"}
            method = call.get("method", "encode")
            embeddings = getattr(st, method)(texts, batch_size=spec.get("batch_size", 32),
                                             convert_to_tensor=True, **kwargs)
            features = st.preprocess(texts, prompt=resolve_prompt(st, method, kwargs))
            outputs = st(dict(features))
            prefix = f"calls/{i + 1}/"
            tensors[prefix + "embeddings"] = embeddings
            tensors[prefix + "input_ids"] = features["input_ids"]
            tensors[prefix + "attention_mask"] = features["attention_mask"]
            if "prompt_length" in features:
                tensors[prefix + "prompt_length"] = torch.tensor(features["prompt_length"])
            tensors[prefix + "token_embeddings"] = outputs["token_embeddings"]
            tensors[prefix + "sentence_embedding"] = outputs["sentence_embedding"]
    meta = {"spec": json.dumps(spec), **st_versions()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out / "acts.safetensors",
              metadata=meta)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {len(st)} modules, {len(spec['calls'])} calls, {size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)


def st_versions():
    import sentence_transformers
    return {"sentence_transformers": sentence_transformers.__version__, **versions()}


def tokenizer(spec):
    """A fast tokenizer of the words "0", "1", ... up to `spec["vocab"] - 1`, whose ids are
    their numbers, followed by `SPECIALS`, split on whitespace."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    n = spec["vocab"]
    vocab = {str(i): i for i in range(n)} | {t: n + i for i, t in enumerate(SPECIALS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if spec.get("lowercase"):
        tok.normalizer = normalizers.Lowercase()
    template = spec.get("template", ["$A"])
    specials = [t for t in template if t in SPECIALS]
    tok.post_processor = processors.TemplateProcessing(
        single=" ".join(template), special_tokens=[(t, vocab[t]) for t in specials])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<|pad|>", unk_token="<|unk|>", cls_token="<|cls|>",
        sep_token="<|sep|>", eos_token="<|eos|>",
        model_max_length=spec.get("model_max_length", 1_000_000),
        padding_side=spec.get("padding_side", "right"))


def resolve_prompt(st, method, kwargs):
    """The prompt that `getattr(st, method)(texts, **kwargs)` prepends to the texts."""
    prompt_name, prompt = kwargs.get("prompt_name"), kwargs.get("prompt")
    if prompt_name is None and prompt is None:
        candidates = {"encode_query": ["query"],
                      "encode_document": ["document", "passage", "corpus"]}.get(method, [])
        prompt_name = next((c for c in candidates if c in st.prompts), None)
    return st._resolve_prompt(prompt, prompt_name)


LEGACY_POOLING_KEYS = {"cls": "pooling_mode_cls_token", "max": "pooling_mode_max_tokens",
                       "mean": "pooling_mode_mean_tokens",
                       "mean_sqrt_len_tokens": "pooling_mode_mean_sqrt_len_tokens",
                       "weightedmean": "pooling_mode_weightedmean_tokens",
                       "lasttoken": "pooling_mode_lasttoken"}


def legacy_configs(out, sentence_bert_config):
    """Rewrite the configurations of a saved model as sentence-transformers < 5 wrote them,
    with the arguments `sentence_bert_config` of its Transformer module, which
    sentence-transformers 5 saves in the tokenizer instead (its `model_max_length` and
    normalizer)."""
    entries = json.loads((out / "modules.json").read_text())
    for entry in entries:
        name = entry["type"].split(".")[-1]
        entry["type"] = f"sentence_transformers.models.{name}"
        folder = out / entry["path"]
        if name == "Pooling":
            config = json.loads((folder / "config.json").read_text())
            modes = config.pop("pooling_mode")
            modes = [modes] if isinstance(modes, str) else modes
            legacy = {"word_embedding_dimension": config.pop("embedding_dimension")}
            legacy.update({key: mode in modes for mode, key in LEGACY_POOLING_KEYS.items()})
            write_json(folder / "config.json", legacy | config)
        elif name == "Normalize":
            (folder / "config.json").unlink(missing_ok=True)
        elif name == "Transformer":
            write_json(folder / "sentence_bert_config.json", sentence_bert_config)
    write_json(out / "modules.json", entries)


def dump_pooling(spec, out, args):
    from sentence_transformers.sentence_transformer.modules import Pooling
    p = spec["pooling"]
    gen = torch.Generator().manual_seed(spec.get("seed", 0))
    lengths, dim = p["lengths"], p["dim"]
    n = max(lengths)
    x = torch.randn(len(lengths), n, dim, generator=gen, dtype=torch.float64)
    positions = torch.arange(n)
    masks = {"right": torch.stack([positions < k for k in lengths]).long(),
             "left": torch.stack([positions >= n - k for k in lengths]).long()}
    tensors = {"token_embeddings": x}
    with torch.no_grad():
        for side, mask in masks.items():
            tensors[f"{side}/attention_mask"] = mask
            for mode in POOLING_MODES + ["all"]:
                for include_prompt in (True, False):
                    pooling = Pooling(dim, pooling_mode=POOLING_MODES if mode == "all" else mode,
                                      include_prompt=include_prompt)
                    features = {"token_embeddings": x, "attention_mask": mask,
                                "prompt_length": p["prompt_length"]}
                    y = pooling(features)["sentence_embedding"]
                    tensors[f"{side}/{mode}/{str(include_prompt).lower()}"] = y
    out.mkdir(parents=True, exist_ok=True)
    meta = {"spec": json.dumps(spec), "modes": json.dumps(POOLING_MODES), **st_versions()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, out / "acts.safetensors",
              metadata=meta)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: pooling, {size / 1e6:.2f} MB")
    return size <= args.max_bytes or bool(args.out)
