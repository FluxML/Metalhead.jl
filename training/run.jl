using Flux, CUDA
CUDA.allowscalar(false)
using Metalhead
using MAT, Images, FileIO
using MLDataPattern, DataLoaders, DataAugmentation
using Statistics: mean
using ParameterSchedulers
using BSON: @save

import LearnBase

include("imagenet.jl")
include("loops.jl")

const DATADIR = "/home/datasets/ILSVRC"
const TRAINDIR = joinpath(DATADIR, "Data/CLS-LOC/train")
const TRAINMETA = joinpath("/home/darsnack/train.txt")
const VALDIR = joinpath(DATADIR, "Data/CLS-LOC/val")
const VALMETA = joinpath("/home/darsnack/val.txt")
const MODELS = [alexnet]

train_dataset = shuffleobs(ImageNet(folder=TRAINDIR, metadata=TRAINMETA))
val_dataset = shuffleobs(ImageNet(folder=VALDIR, metadata=VALMETA))

bs = 128
train_loader = DataLoaders.DataLoader(train_dataset, bs)
val_loader = DataLoaders.DataLoader(val_dataset, bs)

loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
loss(x, y, m) = loss(m(x), y)
accuracy(x, y, m) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))
accuracy(data, m) = mean(accuracy(x, y, m) for (x, y) in data)

for model in MODELS
  @info "Training $model..."
  m = model() |> gpu
  opt = Flux.Optimiser(WeightDecay(1e-4), ADAM(1e-2))
  schedule = ScheduleIterator(Step(λ = opt[2].eta, γ = 0.5, step_sizes = fill(2, 50)))
  cbs = Flux.throttle(() -> @show(accuracy(CuIterator(val_loader), m)), 60*60)
        #  Flux.throttle(() -> (GC.gc(); CUDA.reclaim()), 30)]
  for i in 1:10
    @info "Pass $i / 10..."
    train!(CuIterator(train_loader), m, opt; loss=loss, nepochs=10, schedule=schedule, cb=cbs)
    checkpoint = m |> cpu
    @save "../pretrain-weights/$model.bson" checkpoint
  end
end