# Image datasets

Drop image datasets in this folder for auto-detection support
using `Metalhead.datasets()`. That function will attempt to
auto-detect any standard datasets present in this folder and
make them easily available. This file, lists, for each data
set, instructions for how to obtain them in a format that
is suitable for auto-detection.

# ImageNet

As of 2020, the ImageNet dataset is not openly available.
ImageNet website http://image-net.org makes the images available
only for non-commercial research and/or educational purposes.

To get the dataset go to http://image-net.org/download-images
and sign up for an account. Then you may submit a request. After your request
gets approved then you will be able to download the dataset.

For more information visit http://image-net.org/download-faq

# CIFAR-10

We support the CIFAR-10 dataset in binary format from https://www.cs.toronto.edu/~kriz/cifar.html. Simply download the archive from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz and move the resulting `cifar-10-batches-bin` folder here
for autodetection to support it.
