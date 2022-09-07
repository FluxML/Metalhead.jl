# Layers

Metalhead also defines a module called `Layers` which contains some more modern layers that are not available in Flux. Some of these, like [`StochasticDepth`](@ref) and [`DropBlock`](@ref), are exported from Metalhead as well. Others, like [`conv_norm`](@ref) or [`create_classifier`](@ref) are not exported and are only available by explicitly importing them from `Metalhead.Layers`.

This page contains the API reference for the `Layers` module.

!!! warning

    The `Layers` module is still a work in progress. While we will endeavour to keep the API stable, we cannot guarantee that it will not change in the future. If you find any of the functions in this 
    module do not work as expected, please open an issue on GitHub.

```@autodocs
Modules = [Metalhead.Layers]
```
