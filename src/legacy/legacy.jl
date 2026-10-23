# Hand-written models from Metalhead ≤ 0.9, deprecated in favour of the Hub ports.

# Model utilities
include("core.jl")

# Custom Layers
include("layers/Layers.jl")
include("layers/utilities.jl") # layer utilities
using .Layers

# CNN models
## Builders
include("convnets/builders/invresmodel.jl")
include("convnets/builders/mbconv.jl")
include("convnets/builders/resblocks.jl")
include("convnets/builders/resnet.jl")
include("convnets/builders/stages.jl")
## Older CNN models
include("convnets/alexnet.jl")
include("convnets/vgg.jl")
include("convnets/squeezenet.jl")
## Inceptions
include("convnets/inceptions/googlenet.jl")
include("convnets/inceptions/inceptionv3.jl")
include("convnets/inceptions/inceptionv4.jl")
include("convnets/inceptions/inceptionresnetv2.jl")
include("convnets/inceptions/xception.jl")
## ResNets
include("convnets/resnets/core.jl")
include("convnets/resnets/res2net.jl")
include("convnets/resnets/resnet.jl")
include("convnets/resnets/resnext.jl")
include("convnets/resnets/seresnet.jl")
## DenseNet
include("convnets/densenet.jl")
## EfficientNets
include("convnets/efficientnets/efficientnet.jl")
include("convnets/efficientnets/efficientnetv2.jl")
## MobileNets
include("convnets/mobilenets/mobilenetv1.jl")
include("convnets/mobilenets/mobilenetv2.jl")
include("convnets/mobilenets/mobilenetv3.jl")
include("convnets/mobilenets/mnasnet.jl")
## Others
include("convnets/unet.jl")
## Hybrid models
include("convnets/hybrid/convnext.jl")
include("convnets/hybrid/convmixer.jl")

# Mixers
include("mixers/core.jl")
include("mixers/mlpmixer.jl")
include("mixers/resmlp.jl")
include("mixers/gmlp.jl")

# ViTs
include("vit-based/vit.jl")

# Load pretrained weights
include("pretrain.jl")

# deprecated
include("deprecations.jl")

export AlexNet, VGG, ResNet, WideResNet, ResNeXt, DenseNet,
       GoogLeNet, Inceptionv3, Inceptionv4, InceptionResNetv2, Xception,
       SEResNet, SEResNeXt, Res2Net, Res2NeXt,
       SqueezeNet, MobileNetv1, MobileNetv2, MobileNetv3, MNASNet,
       EfficientNet, EfficientNetv2, ConvMixer, ConvNeXt,
       MLPMixer, ResMLP, gMLP, ViT, UNet

# useful for feature extraction
export backbone, classifier
