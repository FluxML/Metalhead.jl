using Test, Metalhead
using Flux
using Flux: Zygote

function gradtest(model, input)
    y, pb = Zygote.pullback(() -> model(input), Flux.params(model))
    gs = pb(ones(Float32, size(y)))

    # if we make it to here with no error, success!
    return true
end

x_224 = rand(Float32, 224, 224, 3, 1)
x_256 = rand(Float32, 256, 256, 3, 1)

# CNN tests
@testset verbose=true "ConvNets" begin
    include("convnets.jl")
end

GC.safepoint()
GC.gc()

# Other tests
@testset verbose=true "Other" begin
    include("other.jl") 
end

GC.safepoint()
GC.gc()

# ViT tests
@testset verbose=true "ViTs" begin
    include("vit-based.jl") 
end
