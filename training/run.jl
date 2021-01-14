using Flux, CUDA
CUDA.allowscalar(false)
using Metalhead
using MAT, Images, FileIO
using MLDataPattern, DataLoaders, DataAugmentation
using Statistics: mean
using ParameterSchedulers

import LearnBase

include("imagenet.jl")
include("loops.jl")

const DATADIR = "/group/ece/ececompeng/lipasti/libraries/datasets/raw-data"
const MODELS = [alexnet]

train_dataset = shuffleobs(ImageNet(folder=DATADIR))
test_dataset, val_dataset = splitobs(shuffleobs(ImageNet(folder=DATADIR, train=false)); at = 0.05)

bs = 512
train_loader = DataLoaders.DataLoader(train_dataset, bs)
test_loader = DataLoaders.DataLoader(test_dataset, bs)

loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
loss(x, y, m) = loss(m(x), y)
accuracy(x, y, m) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))
accuracy(data, m) = mean(accuracy(x, y, m) for (x, y) in data)

for model in MODELS
  @info "Training $model..."
  m = model() |> gpu
  opt = Flux.Optimiser(WeightDecay(1e-4), Momentum())
  schedule = Exp(λ = 1e-1, γ = 0.9)
  cbs = Flux.throttle(() -> @show(accuracy(CuIterator(test_loader), m)), 240)
        #  Flux.throttle(() -> (GC.gc(); CUDA.reclaim()), 30)]
  train!(CuIterator(train_loader), m, opt; loss=loss, nepochs=1, schedule=schedule, cb=cbs)
end