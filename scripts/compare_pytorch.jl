# Compare Flux model from Metalhead to PyTorch model
# for a sample image
# PyTorch needs to be installed
# Tested on ResNet and VGG models

using Flux
import Metalhead
using DataStructures
using Statistics
using BSON
using PythonCall
using Images
using Test

using MLUtils
using Random

torchvision = pyimport("torchvision")
torch = pyimport("torch")

function jl2np(x::Array)
    x = permutedims(x, ndims(x):-1:1)
    x_np = Py(x).to_numpy()
    return x_np
end

jl2th(x::Array) = torch.from_numpy(jl2np(x))

function np2jl(x::Py)
    x_jl = pyconvert(Array, x) # convert to Any for copyless conversion
    x_jl = permutedims(x_jl, ndims(x_jl):-1:1)
    return x_jl
end

function normalize(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406],(1,1,3,1))
    cstd = reshape(Float32[0.229, 0.224, 0.225],(1,1,3,1))
    return (data .- cmean) ./ cstd
end

# test image
guitar_path = download("https://cdn.pixabay.com/photo/2015/05/07/11/02/guitar-756326_960_720.jpg")

# image net labels
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

function compare_pytorch(jlmodel, pymodel)
    sz = (224, 224)
    img = Images.load(guitar_path);
    img = imresize(img, sz);
    # CHW -> WHC
    data = permutedims(convert(Array{Float32}, channelview(img)), (3,2,1))
    data = normalize(data[:,:,:,1:1])
    
    println("  Flux:")
    Flux.testmode!(jlmodel)
    out = jlmodel(data)
    jlprobs = softmax(out)[:,1]
    for i in sortperm(jlprobs, rev=true)[1:5]
        println("    $(labels[i]): $(jlprobs[i])")
    end

    println("  PyTorch:")
    pymodel.eval()
    out = pymodel(jl2th(data))
    pyprobs = torch.nn.functional.softmax(output[0], dim=0).detach().numpy()
    pyprobs = np2jl(pyprobs)
    for i in sortperm(pyprobs, rev=true)[1:5]
        println("    $(labels[i]): $(pyprobs[i])")
    end


    @test maximum(jlprobs) ≈ maximum(pyprobs)
    @test argmax(jlprobs) ≈ argmax(pyprobs)
    @test sortperm(jlprobs, rev=true)[1:10] ≈ sortperm(pyprobs, rev=true)[1:10] 
    println()
end

jlmodel = Metalhead.VGG(11, pretrain=true)
pymodel = torchvision.models.vgg11(weights="IMAGENET1K_V1")
compare_pytorch(jlmodel, pymodel)
