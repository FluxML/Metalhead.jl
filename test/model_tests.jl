@testsetup module TestModels
using Metalhead, Images, TestImages
using Flux: gradient, gpu
using CUDA: has_cuda

export PRETRAINED_MODELS,
    TEST_FAST,
    _gc,
    gradtest,
    normalize_imagenet,
    TEST_PATH,
    TEST_IMG,
    TEST_X,
    TEST_LBLS,
    acctest,
    x_224,
    x_256,
    gpu,
    has_cuda

const PRETRAINED_MODELS = [
    # (DenseNet, 121),
    # (DenseNet, 161),
    # (DenseNet, 169),
    # (DenseNet, 201),
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

const TEST_FAST = get(ENV, "TEST_FAST", "false") == "true"

function _gc()
    GC.safepoint()
    return GC.gc(true)
end

function gradtest(model, input)
    gradient(sum ∘ model, input)
    # if we make it to here with no error, success!
    return true
end

function normalize_imagenet(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406], (1, 1, 3, 1))
    cstd = reshape(Float32[0.229, 0.224, 0.225], (1, 1, 3, 1))
    return (data .- cmean) ./ cstd
end

# test image
const TEST_IMG = imresize(testimage("monarch_color_256"), (224, 224))
# CHW -> WHC
const TEST_X = let img_array = convert(Array{Float32}, channelview(TEST_IMG))
    permutedims(img_array, (3, 2, 1)) |> normalize_imagenet |> gpu
end

# ImageNet labels
const TEST_LBLS = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

function acctest(model)
    ypred = gpu(model)(TEST_X) |> collect |> vec
    top5 = TEST_LBLS[sortperm(ypred; rev = true)]
    return "monarch" in top5
end

const x_224 = rand(Float32, 224, 224, 3, 1) |> gpu
const x_256 = rand(Float32, 256, 256, 3, 1) |> gpu
end