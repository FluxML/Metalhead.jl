# Contributing to Metalhead.jl

We welcome contributions from anyone to Metalhead.jl! Thank you for taking the time to make our ecosystem better.

You can contribute by fixing bugs, adding new models, or adding pre-trained weights. If you aren't ready to write some code, but you think you found a bug or have a feature request, please [post an issue](https://github.com/FluxML/Metalhead.jl/issues/new/choose).

Before continuing, make sure you read the [FluxML contributing guide](https://github.com/FluxML/Flux.jl/blob/master/CONTRIBUTING.md) for general guidelines and tips.

## Fixing bugs

To fix a bug in Metalhead.jl, you can [open a PR](https://github.com/FluxML/Metalhead.jl/pulls). It would be helpful to file an issue first so that we can confirm the bug.

## Adding models

To add a new model architecture to Metalhead.jl, you can [open a PR](https://github.com/FluxML/Metalhead.jl/pulls). Keep in mind a few guiding principles for how this package is designed:

- reuse layers from Flux as much as possible (e.g. use `Parallel` before defining a `Bottleneck` struct)
- adhere as closely as possible to a reference such as a published paper (i.e. the structure of your model should follow intuitively from the paper)
- use generic functional builders (e.g. [`resnet`](#) is the core function that builds "ResNet-like" models)
- use multiple dispatch to add convenience constructors that wrap your functional builder

When in doubt, just open a PR! We are more than happy to help review your code to help it align with the rest of the library. After adding a model, you might consider adding some pre-trained weights (see below).

## Adding pre-trained weights

To add pre-trained weights for an existing model or new model, you can [open a PR](https://github.com/FluxML/Metalhead.jl/pulls). Below, we describe the steps you should follow to get there.

All Metalhead.jl model artifacts are hosted using HuggingFace. You can find the FluxML account [here](https://huggingface.co/FluxML). This [documentation from HuggingFace](https://huggingface.co/docs/hub/models) will provide you with an introduction to their ModelHub. In short, the Model Hub is a collection of Git repositories, similar to Julia packages on GitHub. This means you can [make a pull request to our HuggingFace repositories](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) to upload updated weight artifacts just like you would make a PR on GitHub to upload code.

1. Train your model or port the weights from another framework.
2. Save the model using [BSON.jl](https://github.com/JuliaIO/BSON.jl) with `BSON.@save "modelname.bson" model`. It is important that your model is saved under the key `model`.
3. Compress the saved model as a tarball using `tar -cvzf modelname.tar.gz modelname.bson`.
4. Obtain the SHAs (see the [Pkg docs](https://pkgdocs.julialang.org/v1/artifacts/#Basic-Usage)). Edit the `Artifacts.toml` file in the Metalhead.jl repository and add entry for your model. You can leave the URL empty for now.
5. Open a PR on Metalhead.jl. Be sure to ping a maintainer (e.g. `@darsnack`) to let us know that you are adding a pre-trained weight. We will create a model repository on HuggingFace if it does not already exist.
6. Open a PR to the [corresponding HuggingFace repo](https://huggingface.co/FluxML). Do this by going to the "Community" tab in the HuggingFace repository. PRs and discussions are shown as the same thing in the HuggingFace web app. You can use your local Git program to make clone the repo and make PRs if you wish. Check out the [guide on PRs to HuggingFace](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) for more information.
7. Copy the download URL for the model file that you added to HuggingFace. Make sure to grab the URL for a specific commit and not for the `main` branch.
8. Update your Metalhead.jl PR by adding the URL to the Artifacts.toml.
9. If the tests pass for your weights, we will merge your PR!

If you want to fix existing weights, then you can follow the same set of steps.
