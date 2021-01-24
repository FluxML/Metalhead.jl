# Training scripts

This folder isn't part of Metalhead.jl. The scripts are used to train the models in Metalhead.jl
to create the pre-trained weight artifacts.

## ImageNet setup

The ImageNet dataset must be manually downloaded and organized in any structure.
`ImageNet(;folder, metadata)` can represent a single dataset (train/test/val), 
and the structure of your ImageNet data folder is embedded into the `metadata` keyword argument.

The `metadata` keyword should point to a file organized as follows:
```txt
<relative path to image> <label index>
<relative path to image> <label index>
...
```

For example we have our data under `/mnt/imagenet/train` and a `/mnt/imagenet/train_labels.txt` metadata file:
```txt
n01440764/n01440764_10026.JPEG 449
n01440764/n01440764_10027.JPEG 449
n01440764/n01440764_10029.JPEG 449
n01440764/n01440764_10040.JPEG 449
...
```
Then we invoke `ImageNet(folder = "/mnt/imagenet/train", metadata = "/mnt/imagenet/train_labels.txt")`.
The ordering of the images will be according to the metadata file.

### Generating metadata

You may find the functions in `training/generate_meta.jl` useful for creating the metadata file. 
For example, if you have an annotations folder and data folder with the same sub-structure 
(common case for ImageNet), then you can do something like:
```julia
julia> include("generate_meta.jl");

julia> classmap = mat2meta("/home/darsnack/meta.mat");

julia> generate_metadata("/home/datasets/ILSVRC/Data/CLS-LOC/train",
                         classmap,
                         "/home/datasets/ILSVRC/Annotations/CLS-LOC/train"),
                         "./train.txt");
```

## Training with single GPU

Once you have prepared the dataset for consumption by `ImageNet`,
you can use the `training/run.jl` script to train a model.
Make sure to set the `MODELS`, `DATADIR`, `TRAINMETA`, and `VALMETA` constants appropriately.
Running the script will look like:
```julia
julia> using CUDA

julia> device!(CUDA.CuDevice(0)) # set a specific GPU if necessary, skip if not

julia> include("run.jl")
```
The script will run 10 passes of 10 epochs each.
It will save a BSON checkpoint to `pretrain-weights/` at the end of each pass.

## Training with multiple GPUs

A distributed training script is provided under `training/run-distributed.jl`.
It is not currently working, but you can test it out with:
```julia
julia> include("run-distributed.jl")
```
Make sure to set the constants correctly as described in the single GPU case.