# Converts the weigths of a PyTorch model to a Flux model from Metalhead
# PyTorch need to be installed
# Tested on ResNet and VGG models

using Flux
using Metalhead
using DataStructures
using Statistics, Random, LinearAlgebra
using BSON
using PythonCall
using Images
using Test

include("utils.jl")

const torch = pyimport("torch")
const torchvision = pyimport("torchvision")

# test image
const GUITAR_PATH = download("https://cdn.pixabay.com/photo/2015/05/07/11/02/guitar-756326_960_720.jpg")
const IMAGENET_LABELS = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

function compare_pytorch(jlmodel, pymodel; rtol = 1e-4)
    sz = (224, 224)
    img = Images.load(GUITAR_PATH);
    img = imresize(img, sz);
    # CHW -> WHC
    data = permutedims(convert(Array{Float32}, channelview(img)), (3,2,1))
    data = imagenet_normalize(data[:,:,:,1:1])
    
    println("Flux:")
    Flux.testmode!(jlmodel)
    out = jlmodel(data)
    jlprobs = softmax(out)[:,1]
    for i in sortperm(jlprobs, rev=true)[1:5]
        println("    $(IMAGENET_LABELS[i]): $(jlprobs[i])")
    end

    println("PyTorch:")
    pymodel.eval()
    out = pymodel(jl2th(data))
    pyprobs = torch.nn.functional.softmax(out[0], dim=0).detach().numpy()
    pyprobs = np2jl(pyprobs)
    for i in sortperm(pyprobs, rev=true)[1:5]
        println("    $(IMAGENET_LABELS[i]): $(pyprobs[i])")
    end

    @test maximum(jlprobs) ≈ maximum(pyprobs) rtol=rtol
    @test sortperm(jlprobs, rev=true)[1:5] == sortperm(pyprobs, rev=true)[1:5]
    println()
end

function _list_state(node::BatchNorm, channel, prefix)
    # use the same order of parameters than PyTorch
    prefix = prefix * ".batchnorm_"
    put!(channel, (prefix * "γ", node.γ)) # weigth (learnable)
    put!(channel, (prefix * "β", node.β)) # bias (learnable)
    put!(channel, (prefix * "μ", node.μ))  # running mean
    put!(channel, (prefix * "σ²", node.σ²)) # running variance
end

function _list_state(node::Conv, channel, prefix)
    prefix = prefix * ".conv_"
    put!(channel, (prefix * "weight", node.weight))
    if node.bias isa AbstractArray
        put!(channel, (prefix * "bias", node.bias))
    end
end

function _list_state(node::Dense, channel, prefix)
    prefix = prefix * ".dense_"
    put!(channel, (prefix * "weight", node.weight))
    if node.bias isa AbstractArray
        put!(channel, (prefix * "bias", node.bias))
    end
end

function _list_state(node::Metalhead.Layers.ClassTokens, channel, prefix)
    put!(channel, (prefix * ".classtoken", node.token)) 
end

function _list_state(node::Metalhead.Layers.ViPosEmbedding, channel, prefix)
    put!(channel, (prefix * ".posembedding", node.vectors))
end

function _list_state(node::LayerNorm, channel, prefix)
    put!(channel, (prefix * ".layernorm_scale", node.diag.scale))
    put!(channel, (prefix * ".layernorm_bias", node.diag.bias))
end

function _list_state(node::Metalhead.Layers.LayerNormV2, channel, prefix)
    put!(channel, (prefix * ".layernorm_scale", node.diag.scale))
    put!(channel, (prefix * ".layernorm_bias", node.diag.bias))
end

function _list_state(node::Metalhead.Layers.MultiHeadSelfAttention, channel, prefix)
    _list_state(node.qkv_layer, channel, prefix * ".qkv")
    _list_state(node.projection, channel, prefix * ".proj")
end

function _list_state(node::Chain, channel, prefix)
    for (i, n) in enumerate(node.layers)
        _list_state(n, channel, prefix * ".layers[$i]")
    end
end

function _list_state(node::SkipConnection, channel, prefix)
    for (i, n) in enumerate(node.layers)
        _list_state(n, channel, prefix * ".layers[$i]")
    end
end

function _list_state(node::Parallel, channel, prefix)
    # reverse to match PyTorch order, see https://github.com/FluxML/Metalhead.jl/issues/228
    for (i, n) in enumerate(reverse(node.layers))
        _list_state(n, channel, prefix * ".parallel[$i]")
    end
end

_list_state(node, channel, prefix) = nothing

function list_state(node; prefix = "model")
    Channel() do channel
        _list_state(node, channel, prefix)
    end
end

function pytorch2flux!(jlmodel, pymodel; verb=false)
    jlstate = OrderedDict(list_state(jlmodel.layers))

    state_dict = pymodel.state_dict()
    pystate = OrderedDict((py2jl(k), th2jl(v)) for (k, v) in state_dict.items() if
                !occursin("num_batches_tracked", py2jl(k)))
   
    jlkeys = collect(keys(jlstate))
    pykeys = collect(keys(pystate))

    ## handle class_token since it is not in the same order
    jl_k = findfirst(k -> occursin("classtoken", k), jlkeys)
    py_k = findfirst(k -> occursin("class_token", k), pykeys)
    if jl_k !== nothing && py_k !== nothing
        jlstate[jlkeys[jl_k]] .= pystate[pykeys[py_k]]
        delete!(pystate, pykeys[py_k])
        delete!(jlstate, jlkeys[jl_k])
    end

    for ((flux_key, flux_param), (pytorch_key, pytorch_param)) in zip(jlstate, pystate)
        println("##")
        @show flux_key size(flux_param) pytorch_key size(pytorch_param)
        @show size(flux_param) == size(pytorch_param)

        param_name = split(flux_key, ".")[end]
        
        if param_name == "dense_weight"
            flux_param .= permutedims(pytorch_param, (2,1))
        elseif  param_name == "conv_weight"
            flux_param .= reverse(pytorch_param, dims=(1, 2))
        else
            flux_param .= pytorch_param
        end
    end
end


## Compare pretrained model to PyTorch
# jlmodel = Metalhead.VGG(11, pretrain=true)
# pymodel = torchvision.models.vgg11(weights="IMAGENET1K_V1")
# compare_pytorch(jlmodel, pymodel)


## Create model, set weights from pytorch, save to BSON 
# modelname = "VGG11"
# weights = "IMAGENET1K_V1"
# jlmodel = VGG(11)
# pymodel = torchvision.models.vgg11(weights=weights)
# pytorch2flux!(jlmodel, pymodel)
# compare_pytorch(jlmodel, pymodel)
# BSON.@save joinpath(@__DIR__,"$(modelname)_$weights.bson") model=jlmodel

