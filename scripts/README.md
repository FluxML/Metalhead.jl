# Scripts

This directory contains scripts that are used to manage the project.

## pytorch2flux.jl

Contains utility function for loading the weights of a pytorch model into the corresponding Flux model.

## port_torchvision.jl

Loads the weights of a selection of torchvision models into Flux models. Relies on `pytorch2flux.jl`.

## manage_huggingface_org.jl

Contains utility functions for managing the [FluxML huggingface organization](https://huggingface.co/FluxML).
Can generate artifacts from saved weights and upload them to HF.
Can also list all the models in the org and create a corresponding `Artifacts.toml` file.
