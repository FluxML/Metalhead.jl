# Metalhead

[![Build Status](https://travis-ci.org/FluxML/Metalhead.jl.svg?branch=master)](https://travis-ci.org/FluxML/Metalhead.jl)

```julia
Pkg.add("Metalhead")
```

This package provides computer vision models that run on top of the [Flux](http://fluxml.github.io/) machine learning library.

![IJulia Screenshot](https://i.imgur.com/spBsaz7.png)

Each model (like `VGG19`) is a Flux layer, so you can do anything you would normally do with a model; like moving it to the GPU, training or freezing components, and extending it to carry out other tasks (such as neural style transfer).

```julia
# Run with dummy image data
julia> x = rand(Float32, 224, 224, 3, 1)
224×224×3×1 Array{Float32,4}:
[:, :, 1, 1] =
 0.353337   0.252493    0.444695   0.767193    …  0.107599   0.424298   0.218889    0.377959
 0.247294   0.039822    0.829367   0.832303       0.582103   0.359319   0.259342    0.12293
  ⋮

julia> vgg(x)
1000×1 Array{Float32,2}:
 0.000851723
 0.00079913
  ⋮

# See the underlying model structure
julia> vgg.layers
Chain(Conv2D((3, 3), 3=>64, NNlib.relu), Conv2D((3, 3), 64=>64, NNlib.relu), Metalhead.#3, Conv2D((3, 3), 64=>128, NNlib.relu), Conv2D((3, 3), 128=>128, NNlib.relu), Metalhead.#4, Conv2D((3, 3), 128=>256, NNlib.relu), Conv2D((3, 3), 256=>256, NNlib.relu), Conv2D((3, 3), 256=>256, NNlib.relu), Conv2D((3, 3), 256=>256, NNlib.relu), Metalhead.#5, Conv2D((3, 3), 256=>512, NNlib.relu), Conv2D((3, 3), 512=>512, NNlib.relu), Conv2D((3, 3), 512=>512, NNlib.relu), Conv2D((3, 3), 512=>512, NNlib.relu), Metalhead.#6, Conv2D((3, 3), 512=>512, NNlib.relu), Conv2D((3, 3), 512=>512, NNlib.relu), Conv2D((3, 3), 512=>512, NNlib.relu), Conv2D((3, 3), 512=>512, NNlib.relu), Metalhead.#7, Metalhead.#8, Dense(25088, 4096, NNlib.relu), Flux.Dropout{Float32}(0.5f0, false), Dense(4096, 4096, NNlib.relu), Flux.Dropout{Float32}(0.5f0, false), Dense(4096, 1000), NNlib.softmax)

# Run the model up to the last convolution/pooling layer
julia> vgg.layers[1:21](x)
7×7×512×1 Array{Float32,4}:
[:, :, 1, 1] =
 0.657502  0.598338  0.594517  0.594425  0.594522  0.597183  0.59534
 0.663341  0.600874  0.596379  0.596292  0.596385  0.598204  0.590494
  ⋮
```

# Working with common datasets
Metalhead includes support for wokring with several common object recognition datasets.
The `datasets()` function will attempt to auto-detect any common dataset placed in
the `datasets/`. The `Metalhead.download` function can be used to download these datasets
(where such automatic download is possible - for other data sets, see `datasets/README.md`),
e.g.:
```
MetalHead.download(CIFAR10)
```

Once a dataset is load, it's training, validation, and test images are available using the
`trainimgs`, `valimgs`, and `testimgs` functions. E.g.

```
julia> valimgs(dataset(ImageNet))[rand(1:50000, 10)]
```

will fetch 10 random validation images from the ImageNet data set.

# Inline Images at the REPL

If you are using OS X, it is recommended that you use iTerm2 and install the
`TerminalExtensions.jl` package. This will allow you to see inference results
as well as the corresponding images directly in your terminal:

![REPL Screenshot](https://i.imgur.com/hy7LXS5.png)
