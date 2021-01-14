# Training scripts

This folder isn't part of Metalhead.jl. The scripts are used to train the models in Metalhead.jl to create the pretrained weight artifacts.

## ImageNet setup

The ImageNet dataset must be manually downloaded and organized in the following structure:
```
root
\__ train
    \__ nXXXXX (image subdirectories, XXXXX is the WNID)
        \__ nXXXXX_YYY.JPEG
        ...
    \__ nXXXXX
    ...
\__ val
    \__ nXXXXX (image subdirectories, XXXXX is the WNID)
        \__ nXXXXX_YYY.JPEG
        ...
    ...
\__ meta.mat (class index to string label map)
```
The `root` directory is specified to constructor as `ImageNet(folder = root)`. You can also specify a different MAT file for the class label metadata with `ImageNet(folder = root, class_metadata = other.mat)` (note this should be relative to `root`).

The `meta.mat` file should be a MAT structured as:
```
"synsets"
\__ "WNID": Vector{String}
\__ "words": Vector{String} (class labels)
\__ "ILSVRC2012_ID": Vector{Int} (class indices)
```

By default, `ImageNet(folder = root)` returns the training dataset. You can use `ImageNet(folder = root, train = false)` to get the validation dataset.