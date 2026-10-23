---
name: port-hf-model
description: Port a timm model from the Hugging Face Hub to Metalhead.jl (Flux), validated numerically against PyTorch. Use when asked to port, add or fix a model in src/Timm/, to write or debug a parity fixture (porting/dump.py, test/fixtures/), or to compare a Flux model with its PyTorch original.
---

# Porting a timm model to Metalhead

A port is a Julia file in `src/Timm/` that mirrors the PyTorch module tree of a timm model.
Weights load by name, with no hand-written mapping. A port is done when its tiny fixtures and
its real-weight references pass. `src/Timm/resnet.jl` is the reference port; read it first.

## Workflow

1. **Inspect.** `cd porting && uv sync && uv run inspect_model.py <arch> [kw=value ...]` prints the
   module tree and the `state_dict`. Quote list values, because zsh globs brackets:
   `'depths=[1,1,1,1]'`. Read the timm source in
   `porting/.venv/lib/python3.12/site-packages/timm/models/<family>.py`: the module classes,
   their `forward`, `forward_features`, `forward_head`, and the `@register_model` entry points.
   Read the constructors in full, including the loops and conditionals. Some modules appear
   only for non-default arguments, such as ConvNeXt's stage-0 downsample when `patch_size == 2`.
2. **Write the port** in `src/Timm/<family>.jl` and include it from `src/Timm/Timm.jl`. Follow
   the conventions below.
3. **Register** every architecture you support with
   `register!(ArchSpec(:timm, "<arch>", builder; defaults = (...), references = [...]))`.
   `defaults` must reproduce what `timm.create_model("<arch>")` builds: the `model_args` of
   the entry point, plus `num_classes = 0` when the default pretrained cfg has no classifier.
   `param_counts.py` prints `num_classes` to check this. Register only architectures whose
   code paths your fixtures cover.
4. **Tiny fixtures.** For each distinct code path (block type, stem type, optional branch),
   write `porting/specs/<name>.json` and run `uv run dump.py specs/<name>.json`. Give every
   keyword the builder accepts a non-default value in at least one spec. Two or three fixtures
   per family are cheap. Then run
   `julia --project=.. check.jl specs/<name>.json` until it prints PASS.
5. **Parameter counts.** Run `uv run param_counts.py <family> <arch> ...` for every registered
   architecture. It writes `test/references/param_counts/<family>.json`. Run this before the
   tests: until then, `check_param_counts` raises a KeyError.
6. **References.** Pick one or two pretrained repos (prefer small ones, and `tv_in1k`/`tv2_in1k`
   when they replace a legacy Metalhead model), then run `uv run dump_reference.py timm/<repo>`.
   Add the printed `repo => sha` to the architecture's `references`. Then run
   `julia --project=.. check_reference.jl ../test/references/timm--<repo>.safetensors`.
   It loads the checkpoint from the Python Hub cache that `dump_reference.py` filled. Correct
   ports land at 1e-7 to 1e-5 in float32, since deeper nets accumulate torch's own float32
   error. To tell that noise from a port error, compare a Julia model with `eltype = Float64`
   against torch in float64 (`timm.create_model('local-dir:' + snapshot, pretrained=True).double()`).
   If both show the same error against the reference, the port is exact.
7. **Tests.** Add `test/models/<family>_tests.jl`, modelled on `test/models/resnet_tests.jl`:
   fixtures, parameter counts, one gradient test, and network-tagged references. Test item
   names must start with `Models: ` (that's how `GROUP=Models` selects them). Make the
   architecture filter (`startswith(arch, "vit_")`) and `reference_files(prefix)` specific
   enough not to catch another family. In Zygote
   gradients, a `Chain`'s entries sit under `.layers` (`g.layer1.layers[1].conv1.weight`).
8. **Run the tests:** `GROUP=Hub` and `GROUP=Models julia --project -e 'using Pkg; Pkg.test()'`.

## Conventions

- **Mirror the module tree.**
  - Each torch submodule attribute is a struct field with the same name.
  - `nn.Sequential` becomes a `Chain` with the same number of entries. Keep parameter-free
    entries (activations, pools, identities) as placeholders, so that Julia index `i` is torch
    index `i-1`.
  - `nn.ModuleList` becomes a `Vector` (`Chain([blocks...])` for sequential stages).
  - `nn.Sequential(OrderedDict(...))` and named submodules of a container become a
    `Chain(name = ...)` with those names.
  - Weight keys are derived from field names, so a renamed field means a missing key.
- **Structs use `@concrete struct Name ... end`** (ConcreteStructs) followed by
  `Flux.@layer Name`. No type annotations on fields, no inner constructors. The parity harness
  rebuilds models with `Functors.functor`.
  - Scalar configuration fields (`eps`, a `Bool` like `use_conv_mlp`, `num_heads`) are fine.
  - Don't define an outer constructor with as many positional arguments as the struct has
    fields: it replaces the default constructor. Name such builders `make_stage(...)`.
  - Struct names are shared by all families in `Metalhead.Timm`. Prefix timm's generic class
    names with the family (`ViTBlock`, `ViTAttention`, `ConvNeXtStage`).
  - Classes from `timm.layers` are ported once, in `src/Timm/layers.jl`, under their timm
    names. Reuse them, and add new ones there: `Mlp`, `LayerNorm2d`, `PatchEmbed`,
    `NormMlpClassifierHead`, `select_adaptive_pool2d`, `create_classifier`,
    `create_conv2d` (with `Conv2dSame` for TF "SAME" padding), `conv_norm_act`
    (`ConvNormAct`), `make_divisible`, `drop_path`, `dropout_or_identity`, `to_2tuple`,
    `avg_pool_ceil`.
  - A torch class that subclasses a torch layer (`Conv2dSame` is an `nn.Conv2d`) has that
    layer's keys. Wrap the Flux layer and forward its keys with
    `layer_params(m::Wrapper) = layer_params(m.conv)`, as `Conv2dSame` does.
- **Use Flux layers and NNlib functions.** Only add a helper to `src/Timm/layers.jl` when no
  combination of them matches the torch module. Prefer a function over a new layer type.
  - `nn.Conv2d` → `Conv`, `nn.Linear` → `Dense`, `nn.BatchNorm2d` → `BatchNorm`,
    `nn.LayerNorm` → `LayerNorm`, `nn.GroupNorm` → `GroupNorm`, `nn.Embedding` → `Embedding`,
    `nn.MaxPool2d` → `MaxPool`, `nn.Dropout` → `Dropout`, `nn.Identity` → `identity`.
  - timm's `BatchNormAct2d` (a BatchNorm2d with `act` and `drop` children and no parameters
    of its own) → `BatchNorm(c, act; eps = 1f-5)`. Flux's `BatchNorm` requires a `Float32`
    `eps`.
  - Activation modules (`nn.ReLU`, `nn.GELU`, ...) become fields holding the NNlib function
    (`relu`, `gelu_erf`, ...). NNlib activations accept whole arrays, so `m.act(x)` works and the
    tap is recorded.
  - Attention: use `NNlib.scaled_dot_product_attention`, with inputs `(head_dim, nheads, seq, batch)`.
    For a fused `qkv` Linear, torch's `qkv(x).reshape(B, N, 3, H, hd)` has its dimensions
    reversed, so no permutation is needed (see `src/Timm/vit.jl`):
    ```julia
    qkv = reshape(m.qkv(x), :, nheads, 3, n, b)
    q, k, v = qkv[:, :, 1, :, :], qkv[:, :, 2, :, :], qkv[:, :, 3, :, :]   # (hd, H, N, B)
    y = reshape(NNlib.scaled_dot_product_attention(q, k, v; fdrop = m.attn_drop), :, n, b)
    ```
    A torch `(B, H, N, hd)` tensor reversed is `(hd, N, H, B)`. It needs
    `permutedims(x, (1, 3, 2, 4))` before the call.
- **Parameters that aren't in a submodule** (`cls_token`, `pos_embed`, layer-scale `gamma`)
  become array fields named like the torch attribute. They load with the torch shape
  reversed, e.g. `(1, 1, D)` becomes `(D, 1, 1)`.
- **Layout.** A Julia array is the torch array with its dimensions reversed: NCHW ↔ WHCN,
  `(B, L, D)` ↔ `(D, L, B)`, `(out, in, kH, kW)` ↔ `(kW, kH, in, out)`. torch `dim = k`
  (0-based, from the front) is Julia `dims = ndims - k`. Since reversal preserves memory order,
  torch `reshape`/`view`/`flatten` become `reshape` with the sizes reversed.
  - `x.flatten(2).transpose(1, 2)` on `(B, C, H, W)` gives `(B, HW, C)`. In Julia, from
    `(W, H, C, B)`: `permutedims(reshape(x, :, C, B), (2, 1, 3))` gives `(C, HW, B)`.
  - `Dense` acts on the first dimension, which is torch's last, so `nn.Linear` on `(B, L, D)`
    is `Dense` on `(D, L, B)` with no reshaping.
  - Spatial hyperparameters are reversed too: torch `kernel_size = (kh, kw)`, `stride` and
    `padding` become Flux `(kw, kh)`. This only matters for non-square values.
- **Weight transforms are automatic** (`src/hub/loadweights.jl`): `Dense` weights are
  transposed, `Conv` kernels are flipped (Flux convolves, torch cross-correlates), and
  everything else is used as stored.
- **Entry points.** Define `forward_features(m::T, x)`, `forward_head(m::T, x; pre_logits = false)`
  and `(m::T)(x) = forward_head(m, forward_features(m, x))`, mirroring timm.
- **Builders** take exactly timm's keyword names and defaults. Error on values the port doesn't
  implement with `check_unsupported`.
  - `model_args` from `config.json` arrive with lists as `Tuple`s and JSON null as `nothing`.
  - Where timm accepts an int or a pair (`to_2tuple`), use `to_2tuple` from `layers.jl`, then
    reverse to Flux order: torch `(h, w)` becomes `(w, h)`.
  - `check_unsupported` rejects any non-default value. An option you only partly support
    needs its own explicit `error`, e.g. `ls_init_value = nothing`.
- **Initialisation** of randomly initialised models should roughly follow timm (e.g.
  `Flux.truncated_normal(std = 0.02)` for ViT Linear layers). Parity tests don't check it.
- **No mutation in forward passes** (no `x[i] = ...`, no `.+=`), so Zygote works. Branch only
  on configuration (`m.downsample === nothing`), never on tensor values.
- **Header.** Every file starts with
  `# Ported from timm.models.<module> (timm <version>, Apache-2.0, Ross Wightman).`

## Numerical pitfalls

- **GELU.** `nn.GELU()` is the exact erf form: use `gelu_erf`. `NNlib.gelu` is the tanh
  approximation, which is `nn.GELU(approximate='tanh')`.
- **Norm epsilon.** Always pass `eps` explicitly. timm often uses `1e-6` (ViT, ConvNeXt
  LayerNorms) while Flux's default is `1f-5`. A wrong eps passes loose checks and fails the
  1e-8 fixture check.
- **LayerNorm over channels** of a WHCN feature map (timm `LayerNorm2d`): use `LayerNorm2d`
  from `layers.jl`. Don't wrap Flux's `LayerNorm` in another struct: that adds a segment to the
  weight keys. For torch's channels-last `nn.LayerNorm` on NHWC, which is CWHN in Julia, use
  Flux's `LayerNorm(C; eps)` directly.
- **Pooling.**
  - `MaxPool` pads with -Inf, like torch.
  - `NNlib.meanpool` has `count_include_pad`, like torch.
  - There is no `ceil_mode`: use `avg_pool_ceil` in `layers.jl`, or pad explicitly.
- **Permutes.** `x.permute(0, 2, 3, 1)` (NCHW to NHWC) is `permutedims(x, (3, 1, 2, 4))`
  (WHCN to CWHN), and `permute(0, 3, 1, 2)` is `permutedims(x, (2, 3, 1, 4))`. On CWHN,
  `Dense` and `LayerNorm` act on the channels, like `nn.Linear` and `nn.LayerNorm` on NHWC.
- **`F.pad`** pads the last dimension first, `(left, right, top, bottom)`. That is the order of
  Julia's leading dimensions (W first).
- **Softmax and reductions.** torch `softmax(dim=-1)` is `softmax(x; dims = 1)`.
  `x.mean((2, 3))` on NCHW is `mean(x; dims = (1, 2))`. `torch.cat(dim = 1)` on NCHW is
  `cat(...; dims = 3)`.
- **`chunk`/`split`** on dim `k` become views along Julia dimension `ndims - k`, keeping the order
  of the chunks.
- **Missing modules.** Where torch has a module (`nn.Identity`, a `DropPath`, `Dropout(0)`),
  store `identity` and call it unconditionally. Store `nothing` only where the torch attribute
  is `None`, and branch on `m.f === nothing`.
- **Dropout and drop-path** are identities in eval mode. Mirror what timm builds:
  - `dropout_or_identity(p)` gives `Dropout(p)`, or `identity` for torch's `nn.Identity()`
    when p = 0.
  - `drop_path(p)` gives a per-sample `Dropout(p; dims = 4)` on WHCN, or `nothing` where timm
    stores `None`.
  - Pass `dims = 3` for `(C, L, B)` tokens.
- **Embedding indices** are 1-based in Julia.

## Fixtures and debugging

- **Spec format:** `{"architecture": "<arch>", "kwargs": {...}, "input_shape": [2, C, H, W]}`.
  `kwargs` shrinks the model (fewer layers and channels, `num_classes = 10`) and is stored as
  the fixture's `model_args`. If `timm.create_model` can't take a kwarg, set
  `"builder": "module:function"`.
- **Size budget.** Keep each fixture under 1 MB (`dump.py` enforces it). The batch is 2, so
  that batch-mixing bugs show up.
- **What `dump.py` does:** it re-randomises every parameter and BN statistic, runs PyTorch in
  float64, and stores in float16 the outputs ("taps") of modules whose names have at most
  one `.` (`blocks.0`, `patch_embed.proj`). `--all-taps` records every module. `check.jl`
  compares:
  - logits and features at 1e-8 (float64);
  - taps at 2e-3, the float16 precision.
- **Debugging subtle errors.** If the logits fail but every tap passes, the error is below
  float16 precision. Dump every module in float32 and compare again:
  `uv run dump.py specs/<name>.json --all-taps --out out/<name>`, then
  `julia --project=.. check.jl out/<name>`. The first failing tap is the first module that
  differs. "not tapped" means the Julia model has no module under that name. That is harmless
  for parameter-free `nn.Identity` placeholders of options the port rejects. Real submodules
  of the supported code path must exist under their torch name.
