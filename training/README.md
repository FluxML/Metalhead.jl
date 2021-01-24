# Training scripts

This folder isn't part of Metalhead.jl. The scripts are used to train the models in Metalhead.jl to create the pre-trained weight artifacts.

## ImageNet setup

The ImageNet dataset must be manually downloaded and organized in any structure. `ImageNet(;folder, metadata)` can represent a single dataset (train/test/val), and the structure of your ImageNet data folder is embedded into the `metadata` keyword argument.

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
Then we invoke `ImageNet(folder = "/mnt/imagenet/train", metadata = "/mnt/imagenet/train_labels.txt")`. The ordering of the images will be according to the metadata file.

### Generating metadata

