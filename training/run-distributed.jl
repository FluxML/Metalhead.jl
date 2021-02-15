using Distributed, CUDA

gpus = [CUDA.CuDevice(5), CUDA.CuDevice(6)]
ngpus = length(gpus)
(nworkers() > 1) && rmprocs.(workers())
addprocs(ngpus)

@everywhere begin
  using Pkg
  Pkg.activate(".")
  using CUDA
end

gpuworkers = asyncmap((zip(workers(), gpus))) do (p, d)
  remotecall_wait(p) do
    device!(d)
    return p
  end
end

@everywhere begin
  CUDA.allowscalar(false)
  using Flux
  using Metalhead
  using MAT, Images, FileIO
  using MLDataPattern, DataLoaders, DataAugmentation
  using Statistics: mean
  using ParameterSchedulers
  using BSON: @save

  import LearnBase

  include("imagenet.jl")
  include("loops.jl")

  loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
  loss(x, y, m) = loss(m(x), y)
  accuracy(x, y, m) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))
  accuracy(data, m) = mean(accuracy(x, y, m) for (x, y) in data)
end

const DATADIR = "/home/datasets/ILSVRC"
const TRAINDIR = joinpath(DATADIR, "Data/CLS-LOC/train")
const TRAINMETA = joinpath("/home/darsnack/train.txt")
const VALDIR = joinpath(DATADIR, "Data/CLS-LOC/val")
const VALMETA = joinpath("/home/darsnack/val.txt")
const MODELS = [alexnet]

train_dataset = shuffleobs(ImageNet(folder=TRAINDIR, metadata=TRAINMETA))
val_dataset = shuffleobs(ImageNet(folder=VALDIR, metadata=VALMETA))

bs = 128 * ngpus
train_loader = DataLoaders.DataLoader(train_dataset, bs)
val_loader = DataLoaders.DataLoader(val_dataset, bs)

@everywhere begin
  ploss(x, y, m) = mean(cpu(pmap(i -> loss(gpu(x[:, :, :, i]), gpu(y[:, i]), m),
                                 Iterators.partition(1:bs, ngpus))))
end

for model in MODELS
  @info "Training $model..."
  @everywhere m = $(model)() |> gpu
  opt = Flux.Optimiser(WeightDecay(1e-4), ADAM())
  schedule = Exp(λ = opt[2].eta, γ = 0.8)
  cbs = Flux.throttle(() -> @show(
                              @sync(@fetchfrom(gpuworkers[1], accuracy(CuIterator(val_loader), m)))),
                      240)
        #  Flux.throttle(() -> (GC.gc(); CUDA.reclaim()), 30)]
  for i in 1:10
    @info "Pass $i / 10..."
    train!(train_loader, m, opt; loss=ploss, nepochs=1, schedule=schedule, cb=cbs)
    @sync @spawnat gpuworkers[1] begin
      checkpoint = m |> cpu
      @save "./pretrain-weights/$model.bson" checkpoint
    end
  end
end