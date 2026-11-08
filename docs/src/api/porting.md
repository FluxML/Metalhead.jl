```@meta
CollapsedDocStrings = true
```

# [Porting](@id porting_api)

```@index
Pages = ["porting.md"]
```

This page is for contributors, and for a package that registers architectures of its own. It
is not public API: the names below are neither exported nor declared `public`, and they may
change without notice, in any release.

Metalhead's libraries (timm, transformers, diffusers and sentence-transformers) plug into
[`create_model`](@ref) through a registry. An
[`AbstractConfigFormat`](@ref Metalhead.AbstractConfigFormat) says how the `config.json` of
a repository names an architecture and its arguments, an
[`ArchSpec`](@ref Metalhead.ArchSpec) is a ported architecture in one format, and
[`register!`](@ref Metalhead.register!) adds it to the registry that `create_model` and
[`list_models`](@ref) read. `src/Timm/config.jl` is the template of a format, and
[Porting a model](@ref contributing) the workflow. The processors have their own interface
(`register_processor!` and the others), listed in the [Internals](@ref internals).

```julia
using Metalhead, Flux
using Metalhead: ArchSpec, AbstractConfigFormat, register!

struct ToyFormat <: AbstractConfigFormat end
Metalhead.architecture_of(::ToyFormat, config) = get(config, "toy_architecture", nothing)

toy_mlp(; in = 3, out = 2) = (; fc = Dense(in => out))   # weight keys follow the field names
register!(ArchSpec(ToyFormat(), "toy_mlp", toy_mlp; family = "toy"))

create_model("toy_mlp"; in = 5)             # a model with initial weights
create_model("path/to/dir"; pretrained = true)  # config.json and model.safetensors in it
```

## Architectures

```@docs
Metalhead.ArchSpec
Metalhead.register!
```

## Config formats

A format is a subtype of `AbstractConfigFormat` with methods of these functions.

```@docs
Metalhead.AbstractConfigFormat
Metalhead.architecture_of
Metalhead.model_args
Metalhead.ignores_config_key
Metalhead.skip_prefix
Metalhead.weights_name
Metalhead.instantiate
```

## Checkpoints

```@docs
Metalhead.Checkpoint
Metalhead.load_state!
```
