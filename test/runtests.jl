using Test, Metalhead
using Flux
using Flux: Zygote
using Images

const PRETRAINED_MODELS = [
    (DenseNet, 121),
    (DenseNet, 161),
    (DenseNet, 169),
    (DenseNet, 201),
    (ResNet, 18),
    (ResNet, 34),
    (ResNet, 50),
    (ResNet, 101),
    (ResNet, 152),
    (ResNeXt, 50, 32, 4),
    (ResNeXt, 101, 64, 4),
    (ResNeXt, 101, 32, 8),
    SqueezeNet,
    (WideResNet, 50),
    (WideResNet, 101),
    (ViT, :base, (16, 16)),
    (ViT, :base, (32, 32)),
    (ViT, :large, (16, 16)),
    (ViT, :large, (32, 32)),
    (VGG, 11, false),
    (VGG, 13, false),
    (VGG, 16, false),
    (VGG, 19, false),
]

function _gc()
    GC.safepoint()
    GC.gc(true)
end

function gradtest(model, input)
    y, pb = Zygote.pullback(() -> model(input), Flux.params(model))
    gs = pb(ones(Float32, size(y)))
    # if we make it to here with no error, success!
    return true
end

function normalize_imagenet(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406], (1, 1, 3, 1))
    cstd = reshape(Float32[0.229, 0.224, 0.225], (1, 1, 3, 1))
    return (data .- cmean) ./ cstd
end

# test image
const TEST_PATH = download("https://cdn.pixabay.com/photo/2015/05/07/11/02/guitar-756326_960_720.jpg")
const TEST_IMG = imresize(Images.load(TEST_PATH), (224, 224))
# CHW -> WHC
const TEST_X = permutedims(convert(Array{Float32}, channelview(TEST_IMG)), (3, 2, 1)) |> normalize_imagenet

# ImageNet labels
const TEST_LBLS = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

function acctest(model)
    ypred = model(TEST_X) |> vec
    top5 = TEST_LBLS[sortperm(ypred; rev = true)]
    return "acoustic guitar" in top5
end

x_224 = rand(Float32, 224, 224, 3, 1)
x_256 = rand(Float32, 256, 256, 3, 1)

# CNN tests
@testset verbose = true "ConvNets" begin
    include("convnets.jl")
end

# Mixer tests
@testset verbose = true "Mixers" begin
    include("mixers.jl")
end

# ViT tests
@testset verbose = true "ViTs" begin
    include("vits.jl")
end
