# Image datasets

Drop image datasets in this folder for auto-detection support
using `Metalhead.datasets()`. That function will attempt to
auto-detect any standard datasets present in this folder and
make them easily available. This file, lists, for each data
set, instructions for how to obtain them in a format that
is suitable for auto-detection.

# ImageNet

As of 2017, the `ILSVRC` competition is being run through Kaggle
and the standard ILSVRC2012 (also known as ImageNet-1k) dataset
is available for download from Kaggle at
https://www.kaggle.com/c/imagenet-object-localization-challenge/data.
Download the file `imagenet_object_localization.tar.gz`, unpack
it and move the resulting `ILSVRC` folder here.

Please note that the file is 154GB, so downloading it may take a while.
Kaggle provides an official command line interface (https://github.com/Kaggle/kaggle-api),
which is able to download the data. However, it does not support download
resumption, which can be problematic for a dataset this size. However, there
is an unofficial API at https://github.com/floydwch/kaggle-cli, which is
able to resume downloads.

# CIFAR-10

We support the CIFAR-10 dataset in binary format from https://www.cs.toronto.edu/~kriz/cifar.html. Simply download the archive from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz and move the resulting `cifar-10-batches-bin` folder here
for autodetection to support it.

# PascalVOC-2012

The zip file needs to be downloaded from https://www.kaggle.com/huanghanchina/pascal-voc-2012/downloads/pascal-voc-2012.zip/1 . unzip the file and move the 'VOC2012' here.
