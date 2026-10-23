# MLPMixer-like models

!!! warning "Deprecated"
    This API is deprecated in Metalhead 0.10 and will be removed in 0.11. See [Migrating from 0.9](@ref migration).

This is the API reference for the MLPMixer-like models supported by Metalhead.jl.

## The higher-level model constructors

```@docs
MLPMixer
ResMLP
gMLP
```

## The core MLPMixer function

```@docs
Metalhead.mlpmixer
```

## The block functions

```@docs
Metalhead.mixerblock
Metalhead.resmixerblock
Metalhead.SpatialGatingUnit
Metalhead.spatialgatingblock
```
