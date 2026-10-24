---
name: port-hf-model
description: Port a timm, transformers or diffusers model from the Hugging Face Hub to Metalhead.jl (Flux), validated numerically against PyTorch. Use when asked to port, add or fix a model in src/Timm/, src/Transformers/ (e.g. a language model such as Llama, Qwen or Mistral) or src/Diffusers/ (a diffusion or flow-matching model, autoencoder, scheduler or text-to-image pipeline such as FLUX.2), to write or debug a parity fixture (porting/dump.py, test/fixtures/), or to compare a Flux model with its PyTorch original.
---

# Porting a timm, transformers or diffusers model to Metalhead

A port is a Julia file in `src/Timm/` that mirrors the PyTorch module tree of a timm model, or
in `src/Transformers/` for a transformers model, or `src/Diffusers/` for a diffusers one (see
the sections at the end, which say what differs).
Weights load by name, with no hand-written mapping. A port is done when its tiny fixtures and
its real-weight references pass. `src/Timm/resnet.jl` is the reference port; read it first.

## Workflow

Most of a port's time goes into waiting for Julia: every new process spends 20–30 s loading
Metalhead before it checks anything. Run Julia in one persistent session instead (see
"Iterating quickly" below).

0. **Set up first, in the background.** Start
   `julia --project=. -e 'using Pkg; Pkg.instantiate(); using Metalhead'` in the background
   (precompiling a fresh checkout takes minutes), and `cd porting && uv sync`. Read the timm
   source meanwhile. In a git worktree, check that `HEAD` has the latest shared helpers in
   `src/Timm/layers.jl` (worktrees may start at `master`); a worktree has no
   `porting/.venv` of its own until its `uv sync`, which takes seconds from uv's cache.
1. **Inspect.** `uv run inspect_model.py <arch> [kw=value ...]` prints the
   module tree and the `state_dict`. Quote list values, because zsh globs brackets:
   `'depths=[1,1,1,1]'`. Read the timm source in
   `porting/.venv/lib/python3.*/site-packages/timm/models/<family>.py`: the module classes,
   their `forward`, `forward_features`, `forward_head`, and the `@register_model` entry points.
   Read the constructors in full, including the loops and conditionals. Some modules appear
   only for non-default arguments, such as ConvNeXt's stage-0 downsample when `patch_size == 2`.
   Check where each helper class lives: only those in `timm/layers/` belong in `layers.jl`.
2. **Write the port** in `src/Timm/<family>.jl` and include it from `src/Timm/Timm.jl`. Follow
   the conventions below. Before defining a helper, search `src/Timm/` for it: two files
   defining the same method fail precompilation ("Method overwriting is not permitted").
3. **Register** every architecture you support with
   `register!(ArchSpec(TIMM, "<arch>", builder; family = "<module>", defaults = (...), references = [...]))`.
   `family` is the timm model module name, used by `list_models(; byfamily = true)` and the docs.
   `defaults` must reproduce what `timm.create_model("<arch>")` builds: the `model_args` of
   the entry point, plus `num_classes = 0` when the default pretrained cfg has no classifier.
   `param_counts.py` prints `num_classes` to check this. Register only architectures whose
   code paths your fixtures cover. Checkpoint tensors of modules the port omits on purpose
   (e.g. Inception's `AuxLogits`) go in `ignore`. An entry point whose arguments change with
   `pretrained = True` (`fbnetc_100` sets `bn_eps` only then) gets the pretrained values in
   `defaults`, since the weights need them; say so in the file header, and pass them
   explicitly in its fixture specs.
4. **Tiny fixtures.** For each distinct code path (block type, stem type, optional branch),
   write `porting/specs/<name>.json` (see "Fixtures" below). Give every
   keyword the builder accepts a non-default value in at least one spec. Two or three fixtures
   per family are cheap. Dump them all with one command,
   `uv run dump.py specs/a.json specs/b.json ...`, and check them in the Julia session with
   `Parity.check_fixture` until all pass (or `julia --project=.. check.jl specs/a.json ...`).
5. **Parameter counts.** Run `uv run param_counts.py <family> <arch> ...` for every registered
   architecture. It writes `test/references/param_counts/<family>.json`. Run this before the
   tests: until then, `check_param_counts` raises a KeyError.
6. **References.** Pick one or two pretrained repos (prefer small ones, and `tv_in1k`/`tv2_in1k`
   when they replace a legacy Metalhead model), then run
   `uv run dump_reference.py timm/<repo> timm/<other> ...`.
   Add the printed `repo => sha` to the architecture's `references`. Then run
   `julia --project=.. check_reference.jl ../test/references/timm--<repo>.safetensors ...`
   from the command line, once for all references (the script exits, so don't `include` it in
   the session).
   It loads the checkpoint from the Python Hub cache that `dump_reference.py` filled. Correct
   ports land at 1e-7 to 1e-5 in float32, since deeper nets accumulate torch's own float32
   error. To tell that noise from a port error, compare a Julia model with `eltype = Float64`
   against torch in float64 (`timm.create_model('local-dir:' + snapshot, pretrained=True).double()`).
   If both show the same error against the reference, the port is exact.
7. **Tests.** Add `test/models/<family>.jl`, modelled on `test/models/resnet.jl`:
   fixtures, parameter counts, a gradient test with `check_gradient`, and the references,
   which download weights and so run only under `TEST_NETWORK &&`. Make the
   architecture filter (`startswith(arch, "vit_")`) and `reference_files(prefix)` specific
   enough not to catch another family. For a transformers model, use
   `reference_files_of("<Class>")`, the files of the registered `references`: one
   organisation's repositories share a prefix across architectures (`Qwen/Qwen3-0.6B`,
   `Qwen/Qwen3-30B-A3B`).
8. **Run the tests** of your family only, by file path prefix:
   `julia --project -e 'using Pkg; Pkg.test(test_args = ["models/<family>"])'` (several are
   allowed, e.g. `["models/mlp_mixer", "models/convmixer"]`). Run all of `["hub", "models"]`
   (several minutes) once, at the end. Don't edit `src/` while a test run is going.

## Iterating quickly

- **Run Julia in a persistent session: the Julia MCP tool** (`mcp__julia__julia_eval`), for all
  Julia code except `Pkg.test`. If it is listed only as a deferred tool, load it first with
  ToolSearch (`select:mcp__julia__julia_eval,mcp__julia__julia_restart`). Pass
  `env_path` = the repository root (your worktree): each `env_path` has its own session. A
  warm session checks a fixture in under a second; a new `julia` process spends 20–30 s
  loading first.
  ```julia
  using Metalhead
  isdefined(Main, :Revise)  # decides `includet` or `include` below
  includet(joinpath(pkgdir(Metalhead), "test", "parity.jl"))  # once per session
  fixture(name) = joinpath(pkgdir(Metalhead), "test", "fixtures", name)
  r = Parity.check_fixture(fixture("<name>")); println(r.pass, " ", r.output)
  ```
  - The tool says it loads Revise, but that needs Revise in the default Julia environment,
    which may lack it. Without Revise, `include` `parity.jl` instead of `includet`, and
    restart (`julia_restart` with the same `env_path`) after each edit to `src/`, then
    reload; a restart and `using Metalhead` take about 15 s. Don't install Revise into the
    global environment yourself.
  - The tool shows only printed output: `println` or `display` the results.
  - Call `Parity.` names qualified, and never `include` `parity.jl` a second time: each include
    makes a new module, and the exported names become ambiguous.
  - With Revise, restart only after changing the fields of a struct, which Revise cannot do,
    or when the session is broken.
- **Without the tool**, batch: `check.jl` and `check_reference.jl` take any number of
  fixtures, and `dump.py` and `dump_reference.py` any number of specs or repos.
- `Pkg.test` builds a new test environment each time (a few minutes): run it for your family
  when the fixtures pass, not as part of the edit loop.
- Keep shell commands simple. Write spec files with your file-writing tool rather than heredocs,
  and put scratch files in a subdirectory of your own (scratch space may be shared).

## Conventions

- **Mirror the module tree.**
  - Each torch submodule attribute is a struct field with the same name.
  - `nn.Sequential` becomes a `Chain` with the same number of entries. Keep parameter-free
    entries (activations, pools, identities) as placeholders, so that Julia index `i` is torch
    index `i-1`.
  - `nn.ModuleList` becomes a `Vector` (`Chain([blocks...])` for sequential stages).
    Never splat many blocks into a Tuple (`Chain(blocks...)`): every architecture then
    compiles its own methods, 5–60 s each (77 EfficientNets took over 20 minutes to build
    once; with Vectors, 40 s).
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
    names. Reuse them, and add new ones there: `Mlp`, `GluMlp`, `GatedMlp`, `LayerNorm2d`,
    `PatchEmbed`, `ClassifierHead`, `NormMlpClassifierHead`, `select_adaptive_pool2d`,
    `create_classifier`, `create_conv2d` (with `Conv2dSame` for TF "SAME" padding),
    `conv_norm_act` (`ConvNormAct`), `SEModule`, `EcaModule`, `create_attn`, `make_divisible`,
    `drop_path`, `token_drop_path`, `dropout_or_identity`, `to_2tuple`.
  - A torch class that subclasses a torch layer (`Conv2dSame` is an `nn.Conv2d`) has that
    layer's keys. Wrap the Flux layer and forward its keys with
    `layer_params(m::Wrapper) = layer_params(m.conv)`, as `Conv2dSame` does.
  - A torch container whose children have no key segment of their own (timm's `MixedConv2d`
    is an `nn.ModuleDict`: `conv_dw.0.weight`) gets a `layer_params` method that returns its
    children's parameters under `"0.weight"`, `"1.weight"`, ... (see `MixedConv2d`).
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
    fdrop = attention_fdrop(m.attn_drop, q)
    y = reshape(NNlib.scaled_dot_product_attention(q, k, v; fdrop), :, n, b)
    ```
    Pass the dropout through `attention_fdrop` (`layers.jl`), which gives `identity` unless
    it is active: with any other `fdrop`, NNlib can't take its fused implementation, which is
    3 times faster in `Float32` and 50 times in `BFloat16` on a GPU.
    A torch `(B, H, N, hd)` tensor reversed is `(hd, N, H, B)`. It needs
    `permutedims(x, (1, 3, 2, 4))` before the call. Multi-query attention (one key/value
    head shared by all query heads, MobileNetV4) is the call with keys and values of 1 head
    (`MultiQueryAttention2d` in `layers.jl`).
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
  Where timm defines `forward_intermediates`, port it and `feature_info` too, with the helpers
  of `src/Timm/features.jl` (ResNet's is the reference). `dump.py` records its outputs, and
  `check_fixture` fails until the Julia method matches them.
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
  approximation, which is `nn.GELU(approximate='tanh')`. CLIP's QuickGELU is `quick_gelu`.
- **Norm epsilon.** Always pass `eps` explicitly. timm often uses `1e-6` (ViT, ConvNeXt
  LayerNorms) while Flux's default is `1f-5`. A wrong eps passes loose checks and fails the
  1e-8 fixture check. Check each norm separately: a block may set one eps and create a
  submodule with torch's default (gMLP's `SpatialGatingUnit` uses `nn.LayerNorm`'s 1e-5 inside
  a block whose norms use 1e-6).
- **LayerNorm over channels** of a WHCN feature map (timm `LayerNorm2d`): use `LayerNorm2d`
  from `layers.jl`. Don't wrap Flux's `LayerNorm` in another struct: that adds a segment to the
  weight keys. For torch's channels-last `nn.LayerNorm` on NHWC, which is CWHN in Julia, use
  Flux's `LayerNorm(C; eps)` directly.
- **Pooling.**
  - `MaxPool` pads with -Inf, like torch.
  - `NNlib.meanpool` has `count_include_pad`, like torch.
  - `NNlib.maxpool` and `NNlib.meanpool` take `ceil_mode`, like torch, but Flux's `MaxPool`
    and `MeanPool` don't: call the NNlib function in a closure.
- **Resizing.** torch's `F.interpolate(mode="bicubic")` is `NNlib.upsample_bicubic`, with
  `align_corners` and `antialias`. It defaults to `align_corners = false`, as torch does, but
  `upsample_bilinear` defaults to `true`.
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
- **Raw kernels.** Only `Conv` weights are flipped on load. A kernel kept as a plain array
  (CondConv2d's per-expert kernels, mixed per sample) is torch's, so apply it with
  `NNlib.conv(x, w; flipped = true)`, torch's cross-correlation.

## Julia and Flux pitfalls

- `chunk` comes from MLUtils (imported in `src/Timm/Timm.jl`), not Flux. There is no
  `NNlib.global_meanpool`: use `GlobalMeanPool()(x)`. `mean` is imported in `Timm.jl`.
- `BatchNorm` needs a `Float32` `eps`, the type of its `momentum`.
- `Flux.paramtype` is gone in Flux 0.16. Convert precision with `Metalhead.ELTYPE_CONVERTERS[T]`
  (or `f32`/`f64`).
- `Conv` has 8 fields, `pad_mode` last, and `BatchNorm` stores its channel count in `chs`.
  Construct layers with their public constructors rather than positionally where possible.
- Constant arrays built in the forward pass (blur filters, pooling matrices) must not be
  mutated, or Zygote fails: build them with comprehensions and mark the builder
  `ChainRulesCore.@non_differentiable`. Broadcasting functions like `binomial.` also breaks
  under Zygote.
- Zygote can't differentiate `cumsum` of a Tuple of Ints (it fails with
  `length(::ZeroTangent)`): compute such channel ranges in a helper marked
  `ChainRulesCore.@non_differentiable` (`split_ranges` in `layers.jl`).
- Each test file runs in a module of its own, so helpers shared by several files go in
  `test/parity_setup.jl`, exported. Statistics is not in the test environment.
- Zygote gradients mirror the model, with `Chain` entries under `.layers`, `LayerNorm` weights
  under `.diag.scale` and a `Conv2dSame`'s under `.conv.weight`. `check_gradient(model, x)`
  avoids writing these paths: it checks that every trainable array gets a finite gradient,
  which also catches submodules the forward pass never calls.

## Half precision

`create_model(...; eltype = Float16)` and `f16`/`bf16` convert in mixed precision: the
normalization layers keep their parameters and statistics in `Float32` and normalise in
`Float32`, every other array becomes 16-bit, and so does the data flowing through.

- **A norm layer you write** (`RMSNorm`, `LayerNorm2d`) opts out of the cast with
  `Flux.f16(m::MyNorm, cx) = f32(m, cx)` and `Flux.bf16(m::MyNorm, cx) = f32(m, cx)` after
  its `Flux.@layer`, as Flux's `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` do,
  and passes its parameters to NNlib: `NNlib.rmsnorm(m.weight, x; eps)`, or
  `NNlib.layernorm(g, b, x; dims, eps)` with `g` and `b` reshaped to broadcast against `x`.
  These return the eltype of `x`, and throw on 16-bit parameters with 16-bit input. Don't
  multiply by the weight yourself: a `Float32` weight times `Float16` data is `Float32`.
- Other parameters (layer scale `gamma`, ResMLP's `Affine`) convert like any weight.
- **Constants built in the forward pass** (blur filters, pooling matrices, rotary tables,
  masks) take `eltype(x)`, and scalars `oftype(x, c)` or `eltype(x)(c)`: a `Float32` array
  promotes the output to `Float32`, and a Float64 literal (`1.702 .* x`) to Float64. Where
  the original computes in float32 for half-precision data (timm's position-embedding
  resampling), compute in `promote_type(eltype(x), Float32)` and convert the result back.
- **Test** with `check_half_precision(name)` (`test/parity_setup.jl`): it converts a
  fixture to `Float16` and checks the eltypes of the norm parameters, of the other arrays
  and of the output, that no warning is logged, and the error against `Float32` (1e-2).
  Never run `BFloat16` on the CPU in the tests, only on a GPU: CPU matrix products and
  convolutions are slow and accumulate in 16 bits. A test may check the eltypes that
  `eltype = BFloat16` gives, without running the model.

## Fixtures and debugging

- **Spec format:** `{"architecture": "<arch>", "kwargs": {...}, "input_shape": [2, C, H, W]}`.
  `kwargs` shrinks the model (fewer layers and channels, `num_classes = 10`) and is stored as
  the fixture's `model_args`. If `timm.create_model` can't take a kwarg, set
  `"builder": "module:function"`. Many entry points hard-code their `model_args`, so that
  `timm.create_model("convmixer_768_32", dim = 16)` fails with "got multiple values for keyword
  argument": build the model class directly (a small builder in `porting/`, e.g.
  `porting/mixers.py`). The fixture's `model_args` must be valid Julia builder kwargs, so pass
  classes and partials (`norm_layer = partial(nn.BatchNorm2d, eps = 1e-3)`) by name or by the
  value they set (`norm_eps`), and note any such non-timm kwarg in the builder's docstring.
- **Size budget.** Keep each fixture under 1 MB (`dump.py` enforces it). The batch is 2, so
  that batch-mixing bugs show up. The widest stage and the stem dominate the size: use a stem
  width of about 8 and 4–8 channels per stage. With `block_reduce_first`, cardinality and
  similar options, check that the shrunk widths stay divisible.
- **Fixed-width models** (no width option, like Inception) shrink with `"shrink": k`: every
  channel count except the image channels and `num_classes` is divided by `k` on both sides
  (`shrunk_channels` in `porting/common.py`, `shrink_channels` in `test/parity.jl`), and
  `fixture_model` builds the full-size Julia model, shrinks it and loads the weights strictly.
  Every other channel count must be divisible by `k` (multiples of 8 from `round_channels`
  allow `k = 8`). Only depthwise `groups` are rescaled: other convs keep their number of
  groups, so the shrunk channels must stay divisible by it. `shrink` patches the
  constructors of torch layers, so it can't shrink a module that splits channels by its own
  full-size count (`MixedConv2d`) or owns parameters (`CondConv2d`): use a width multiplier. `shrink` also suits generators that hard-code a wide head (MobileNetV3's
  1280 features), and it builds the registered architecture at full size, so the fixture
  checks the registered `defaults` too. `"weight_dtype": "float16"` halves the
  weights (PyTorch then uses the rounded values, so the check stays exact), and
  `"tap_depth": d` taps modules with at most `d` dots (default 1; `-1` records no taps).
  `shrink` only rescales Conv, BatchNorm and Linear layers, so it fails on modules that
  hold a width elsewhere: attention head dims (`MultiQueryAttention2d`'s `key_dim`), layer
  scale `gamma`s and `BlurPool2d` (its `channels`). Shrink such models with a small
  `channel_multiplier` through a generator builder instead (MobileNetV4's hybrid and blur
  fixtures).
- **Taps of the stem** are often the largest: a full-resolution `BatchNormAct2d` records
  three taps (`bn1`, `bn1.drop`, `bn1.act`). With a fixed-width stem, lower the input's
  channels and size, or `tap_depth`, before the model's width.
- **Generator builders.** A `"builder"` spec calling timm's `_gen_*` passes its `variant`
  argument, which then lands in the fixture's `model_args`: the Julia `gen_*` must accept
  `variant`, and use it where timm reads it (MobileNetV3's `"small"`, `"minimal"`).
- **Attention fixtures** need several tokens at every attention block: with one key token
  the softmax is constant and the query and key weights go unchecked. Pick an input size
  that leaves at least 2×2 positions at the last attention stage.
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
  for parameter-free `nn.Identity` placeholders of options the port rejects, and for the
  `drop` and `act` children of timm's `BatchNormAct2d`, which Flux's `BatchNorm` holds inside.
  Real submodules of the supported code path must exist under their torch name.

## transformers models

`src/Transformers/llama.jl` is the reference port; read it and `src/Transformers/layers.jl`
first. The workflow is the one above, with these differences.

- **Source.** Read `porting/.venv/lib/python3.*/site-packages/transformers/models/<model>/`
  `modeling_<model>.py` and `configuration_<model>.py`. Many models are Llama with a few
  changes (`# Copied from` or `modular_<model>.py` say which): reuse `RMSNorm`,
  `RotaryEmbedding` and `rotary_tables` (for `NNlib.apply_rotary_embedding`), `TiedLMHead`
  and `rope_parameters_of` from `layers.jl`, and add shared blocks there.
- **Llama subclasses.** Where `modular_<model>.py` subclasses the Llama classes, the port
  reuses the Llama structs, as Res2Net reuses timm's `ResNet`: its builder passes
  `llama_like` (`llama.jl`) a function that builds the attention of a layer, and defines only
  the modules that differ. `qwen2.jl` changes only which projections have a bias;
  `qwen3.jl` adds a `Qwen3Attention` with per-head `q_norm` and `k_norm`. A model whose
  layers differ more but whose decoder runs like Llama's (`gemma3.jl`) builds a `LlamaModel`
  of its own layers, and a `...ForCausalLM <: CausalLM` where its head differs. A model that
  changes the embeddings outside the embedding module (Granite's `embedding_multiplier`) is a
  `DecoderModel` of its own with an `embed` method (`GraniteModel`): folding the change into
  the embedding would fail the tap of `embed_tokens`. Weight keys come
  from field names, so the struct names don't matter for loading. `attention_heads` derives
  `head_dim` and `num_key_value_heads`; an explicit `head_dim` need not divide
  `hidden_size` (Qwen3-0.6B: 16 heads of 128 features, hidden size 1024).
- **Derived config fields.** A config class's `__post_init__` can derive fields from others,
  and `save_pretrained` writes the derived values to `config.json`. Qwen2 and Qwen3 set
  `sliding_window = None` unless `use_sliding_window`, then derive `layer_types` from
  `max_window_layers`. Declare the derived field (`layer_types`) as well as its sources,
  compute it the same way when it is `nothing`, and check the result: `check_full_attention`
  in `layers.jl` does this for the sliding-window attention of Qwen2 and Qwen3, which their
  ports don't implement, and `gemma3_layer_types` for Gemma 3's, which its port does.
  Probe edge cases in Python before choosing what to reject (`uv run python -I -c ...`):
  transformers itself fails on a `"sliding_attention"` layer when `use_sliding_window` is false.
- **Undeclared and validated fields.** Grep the modeling file for `getattr(config, ...)`: it
  reads keys that the config class may not declare but a `config.json` can carry (Qwen2's
  `head_dim`). Declare those too, or `model_args` drops them. The config's validators
  (`validate_architecture`) say what to reject: `LlamaConfig` requires `hidden_size`
  divisible by `num_attention_heads` even with an explicit `head_dim`; Qwen3's doesn't.
- **Registry.** Register the class name with the transformers `ConfigFormat`
  (`src/Transformers/config.jl`) and its model module as the family, e.g.
  `register!(ArchSpec(TRANSFORMERS, "Qwen2ForCausalLM", builder; family = "qwen2", references = [...]))`.
  `create_model` looks it up from `config.json`'s `architectures`. The builder takes the
  fields of the config class as keywords, with their names and defaults. `model_args` passes
  it only the keys of `config.json` it declares, as transformers ignores the generic ones
  (token ids, `transformers_version`, ...), so declare every field the model reads. Older
  checkpoints have `rope_theta` and `rope_scaling` where transformers 5 writes
  `rope_parameters`: accept all three (`rope_parameters_of`). It supports the `rope_type`s
  `"default"`, `"linear"` and `"llama3"`; `inv_freq` (`layers.jl`) computes their frequencies,
  and `_exact_inv_freq` (`porting/common.py`) the same in float64 for the fixtures, checked
  against transformers' own. A model with a rotary embedding for each layer type (Gemma 3)
  holds a NamedTuple of `RotaryEmbedding`s, and its attention picks the tables of its type.
- **No parameter counts.** A transformers configuration is a checkpoint, not a named
  architecture, so the registry has no defaults to check. Strict loading of the reference
  checks the real configuration.
- **Layout.** Token ids `(B, L)` become indices `(L, B)`, 1-based (`dump.py` and the harness
  add 1). Hidden states are `(D, L, B)`, so `nn.Linear` is `Dense` with no reshaping, and
  `view(B, L, H, hd)` is `reshape(x, hd, H, L, B)`, the layout of
  `NNlib.scaled_dot_product_attention` (use `is_causal = true`; it repeats key/value heads for
  grouped-query attention like transformers' `repeat_kv`). `nn.ModuleList` is a plain
  `Vector` when its layers take extra arguments (the rotary tables).
- **Tied embeddings.** With `tie_word_embeddings`, `lm_head` is a `TiedLMHead` holding the
  `Embedding`'s own weight array. `torch_params` then lists `lm_head.weight` as an alias, which
  checkpoints usually omit, and `f16`/`gpu` keep the tie.
- **Buffers.** Non-persistent buffers (`inv_freq`) have no `state_dict` key, so they must not be
  array fields: compute them in the forward pass inside `ChainRulesCore.ignore_derivatives`,
  on the device of the input.
- **Float32 inside float64.** transformers computes RMSNorm and the rotary tables in float32
  even in a float64 model, and its eager attention takes the softmax in float32. The fixtures
  use `attn_implementation="sdpa"` and `exact_float64` (`porting/common.py`) to compute in
  float64; patch any new float32 cast the same way, or the fixture misses 1e-8. A float32
  buffer that holds a constant (Gemma 3's `embed_scale`, `√hidden_size`) stays rounded to
  float32 in a float64 model: round it to `Float32` in Julia too.
- **Sliding-window attention** (Gemma 3): `sliding_window_mask` is the window's `mask` for
  `NNlib.scaled_dot_product_attention`, on top of `is_causal`. In a batch padded on the right,
  a padding query whose window holds only padding attends to nothing: PyTorch returns zeros
  there, and NNlib NaNs unless `sliding_window_q_lengths` gives it the matching `q_lengths`.
  Give a padded fixture such queries (Gemma 3's `lengths = [9, 7, 2]` with a window of 4).
- **Mixtures of experts** (Granite MoE). transformers 5 stores the experts of a layer as
  stacked 3D parameters (`experts.gate_up_proj`, `experts.down_proj`) and renames or merges
  the checkpoint's keys when loading (`conversion_mapping.py`); `save_pretrained` writes the
  checkpoint's keys back, so fixtures have them too. Mirror the checkpoint's keys: Granite's
  `input_linear.weight` is a plain 3D array field, `(in, out, num_experts)` in Julia. Compute
  the experts with `moe_experts` (`layers.jl`), and pick them with `MLUtils.topk`, which runs
  on the GPU too. Routers often compute in float32 (`.float()`, `softmax(dtype=torch.float)`):
  patch them in `exact_float64`, and in Julia compute the routing weights in
  `promote_type(eltype(x), Float32)`. `build_transformers` uses the eager experts
  (`experts_implementation="eager"`), and `rerandomize` gives stacked `..._proj` parameters
  the scale of their fan-in. Avoid `NNlib.scatter` in forward passes: its atomic additions
  are slow in `BFloat16` on a GPU (100 times slower than in `Float32` on an RTX 5090). Test the
  fixtures' sensitivity by changing the router (softmax over all experts, lowest experts):
  with tiny models, a wrong router can still match to a few percent.
  `bfloat16` checkpoints load into Float32 arrays without any special handling.
- **Specs.** `{"library": "transformers", "architecture": "LlamaForCausalLM",
  "model_type": "llama", "config": {...}, "input_shape": [B, L], "tap_depth": 3}`. `dump.py`
  writes the fixture with `save_pretrained`, and taps the modules that have parameters
  (`model.layers.0.self_attn`, `...mlp`, the norms). Cover each option of the config (bias,
  `head_dim`, grouped and plain attention, tied and untied head) in at least one spec, and
  error on the others (`rope_type`, `hidden_act`).
- **References.** `uv run dump_reference.py <org>/<repo>` tokenizes `PROMPTS` with the repo's
  tokenizer and stores the ids, the hidden states, the last-position logits and the greedy
  continuation (`check_lm_reference` in `test/parity.jl`); the logits come from the model's
  own head, which may scale (Granite) or softcap them. Prefer the smallest checkpoint of
  the architecture. Tokenizers are not part of Metalhead; HuggingFaceTokenizers.jl loads
  them for end-to-end checks outside the test suite.

## diffusers models and pipelines

`src/Diffusers/transformer_flux2.jl` and `src/Diffusers/autoencoder_kl_flux2.jl` are the
reference ports, and `pipeline_flux2_klein.jl` the reference pipeline. The workflow is the one
above, with these differences.

- **Source.** Read `porting/.venv/lib/python3.*/site-packages/diffusers/models/<kind>/<file>.py`
  (`transformers/`, `autoencoders/`) and `pipelines/<name>/pipeline_<name>.py`. Shared blocks
  are in `models/embeddings.py`, `normalization.py`, `attention_processor.py`, `resnet.py`,
  `unets/unet_2d_blocks.py` and `autoencoders/vae.py`: port them once, in
  `src/Diffusers/layers.jl` or `vae.jl`. Leave out what text-to-image doesn't run (the KV cache
  of reference images, LoRA, fused projections, tiling) and say so in the file header.
- **Registry.** Register the class of `_class_name` with `DIFFUSERS` and the pipeline family,
  e.g. `register!(ArchSpec(DIFFUSERS, "AutoencoderKLFlux2", builder; family = "flux2", references = [...]))`.
  The builder takes the arguments of the class's `__init__` (`@register_to_config`), with their
  names and defaults; `model_args` passes it the keys of `config.json` it declares. Weights are
  `diffusion_pytorch_model.safetensors` (`weights_name`). No parameter counts, as for
  transformers.
- **Pipelines** register their class with `DIFFUSERS_PIPELINE` (identified by the `...Pipeline`
  suffix of `_class_name`). `instantiate` (`src/Diffusers/config.jl`) builds each component of
  `model_index.json` from its subfolder: a model from its `config.json`, a scheduler from
  `scheduler_config.json` through `SCHEDULERS` (class name => builder), a tokenizer as the
  directory of its files (the HuggingFaceTokenizers.jl extension in `ext/` tokenizes). The
  builder takes the components and the other declared keys (`is_distilled`). Schedulers keep
  no state: port `set_timesteps` and `step` as functions of the noise levels.
- **Layout.** Images are WHCN; tokens `(C, L, B)`; position ids `(n_axes, L)` integers (torch
  `(B, L, n_axes)`, of which diffusers takes the first); timesteps vectors. The Julia forward
  takes the positional arguments of the torch `forward`, in its order, so that fixtures call it
  with `inputs/<i>`.
- **Float32 inside float64.** diffusers computes `get_timestep_embedding`, the rotary tables
  (`.float()` in `get_1d_rotary_pos_embed`), `apply_rotary_emb` (`x.float()`) and the Euler
  `step` of the scheduler in float32. `exact_float64_diffusers` (`porting/common.py`) patches
  them; a model that imports a helper by name (`from ..embeddings import apply_rotary_emb`) needs
  the patch in its own module. Pipelines also keep the noise levels and the eps of the latent
  statistics in float32, so pipeline fixtures match to about 1e-12.
- **Half precision** (see "Half precision" above). diffusers casts to float32 where 16 bits
  lose too much: the sinusoidal timestep embeddings (angles of timesteps near 1000 are off by
  radians in BFloat16) and the scheduler's step. Do the same, and convert back like torch's
  `.to(x.dtype)` (`convert_like`). Statistics that stay `Float32` (the latent `BatchNorm` of a
  FLUX.2 autoencoder) promote 16-bit data when used by hand: convert the result. Where
  diffusers does compute in 16 bits (the transformer scales the timesteps by 1000 in the
  model's eltype), mirror it, and loosen `check_half_precision` for that fixture with a
  comment, rather than diverge from diffusers.
- **Specs.** `{"library": "diffusers", "architecture": "<class>", "config": {...}, "inputs": {...}, "tap_depth": 2}`.
  `diffusers_inputs` in `porting/dump.py` builds the inputs of each class from the sizes in
  `"inputs"`; add a branch for a new class. A pipeline spec has `"components"` (configs, with
  the text encoder as a transformers spec) and the settings of a run (`porting/specs/flux2_klein_tiny.json`);
  `dump_flux2_klein` saves the pipeline without a tokenizer and runs it from random token ids
  and fixed noise, and `Parity.check_pipeline_fixture` compares the prompt embeddings, latents
  and images. Give a tiny text encoder fewer layers than `text_encoder_out_layers` asks for
  only knowingly: transformers norms the last of `output_hidden_states`.
- **Taps.** torch hooks modules that diffusers calls as modules even when they have no
  parameters (`pos_embed`); a Julia helper that computes the same tables leaves them
  "not tapped", which is harmless. A tapped submodule is wrapped in a `Tap`, so helpers that
  take it must not dispatch on its type (`rotary_tables(m, ids, x)`).
- **References.** `uv run dump_reference.py <org>/<repo>/<subfolder>` for a component (an
  autoencoder or a transformer; add a branch in `dump_diffusers_reference` for another class),
  and `<org>/<repo>` for a pipeline (`dump_pipeline_reference`). Random inputs are stored
  rounded to float16, so both sides use the same values. Register the reference of a component
  as `"<org>/<repo>/<subfolder>" => sha`: `create_model` reads `org/name/subfolder` as a
  subfolder. Over several denoising steps float32 errors grow, from about 1e-5 for one step;
  to tell that from a port error, run both pipelines in float64 with the same embeddings and
  noise. Tests of multi-gigabyte references run only with `TEST_LARGE`
  (`METALHEAD_TEST_LARGE=true`).
