# [Porting API](@id porting_api)

Metalhead's libraries (timm, transformers, diffusers and sentence-transformers) plug into
[`create_model`](@ref) through a registry. An
[`AbstractConfigFormat`](@ref Metalhead.AbstractConfigFormat) says how the `config.json` of
a repository names an architecture and its arguments, an
[`ArchSpec`](@ref Metalhead.ArchSpec) is a ported architecture in one format, and
[`register!`](@ref Metalhead.register!) adds it to the registry that `create_model` and
[`list_models`](@ref) read. `src/Timm/config.jl` is the template of a format, and
[Porting a model](@ref contributing) the workflow.

This API is public, and Metalhead is 0.x: it may change in a minor release, which the
[changelog](https://github.com/FluxML/Metalhead.jl/blob/master/CHANGELOG.md) then lists. The
interface of processors (`register_processor!` and the others of [Porting internals](@ref))
is not public yet.

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
Metalhead.skip_prefix
Metalhead.weights_name
Metalhead.instantiate
```

## Checkpoints

```@docs
Metalhead.Checkpoint
Metalhead.load_state!
```
