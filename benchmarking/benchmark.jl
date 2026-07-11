
using Colors
using CUDA, cuDNN
using DataFrames
using Flux
using Flux: logitcrossentropy, onecold, onehotbatch
using GLMakie
using Metalhead
using MLDatasets
using Optimisers
using ProgressMeter
using TimerOutputs

include("tooling.jl")

function run()

epochs = 45
batchsize = 1000
device = gpu
CUDA.allowscalar(false)
allow_skips = true

train_loader, test_loader, labels = load_cifar10(; batchsize)
nlabels = length(labels)
firstbatch = first(first(train_loader))
imsize = size(firstbatch)[1:2]

@info "Benchmarking" epochs batchsize device imsize

to = TimerOutput()

common = "pretrain=false, inchannels=3, nclasses=$(length(labels))"

# these should all be the smallest variant of each that is tested in `/test`
modelstrings = (
    "AlexNet(; $common)",
    "VGG(11, batchnorm=true; $common)",
    "SqueezeNet(; $common)",
    "ResNet(18; $common)",
    "WideResNet(50; $common)",
    "ResNeXt(50, cardinality=32, base_width=4; $common)",
    "SEResNet(18; $common)",
    "SEResNeXt(50, cardinality=32, base_width=4; $common)",
    "Res2Net(50, base_width=26, scale=4; $common)",
    "Res2NeXt(50; $common)",
    "GoogLeNet(batchnorm=true; $common)",
    "DenseNet(121; $common)",
    "Inceptionv3(; $common)",
    "Inceptionv4(; $common)",
    "InceptionResNetv2(; $common)",
    "Xception(; $common)",
    "MobileNetv1(0.5; $common)",
    "MobileNetv2(0.5; $common)",
    "MobileNetv3(:small, width_mult=0.5; $common)",
    "MNASNet(:A1, width_mult=0.5; $common)",
    "EfficientNet(:b0; $common)",
    "EfficientNetv2(:small; $common)",
    "ConvMixer(:small; $common)",
    "ConvNeXt(:small; $common)",
    # "MLPMixer(; $common)", # no tests found
    # "ResMLP(; $common)", # no tests found
    # "gMLP(; $common)", # no tests found
    "ViT(:tiny; $common)",
    # "UNet(; $common)" # doesn't support kwargs "inchannels", "nclasses"
    )
df = DataFrame(; model=String[], train_loss=Float64[], train_acc=Float64[], test_loss=Float64[], test_acc=Float64[])
cols = distinguishable_colors(length(modelstrings), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
f = Figure()
ax = Axis(f[1, 1], title="CIFAR-10 Training on a Nvidia 3090, batch 1000\nTest accuracy vs. time over 45 epochs", xlabel="Time (s)", ylabel="Testset Accuracy")
display(f)
max_x = 0
for (i, modstring) in enumerate(modelstrings)
    @timeit to "$modstring" begin
        @info "Evaluating $i/$(length(modelstrings)): $modstring"
        # Initial precompile is variable based on what came before, so don't time first load
        eval(Meta.parse(modstring))
        # second load simulates what might be possible with a proper set-up pkgimage workload
        @timeit to "Load" model=eval(Meta.parse(modstring))
        @timeit to "Training" ret = train(model,
            train_loader,
            test_loader;
            limit = 1,
            to,
            device
            )
        elapsed, train_loss, train_acc, test_loss, test_acc = if isnothing(ret)
            allow_skips || break
            missing, missing, missing, missing, missing
        else
            elapsed, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = ret
            max_x = max(maximum(elapsed), max_x)
            lines!(ax, elapsed, test_acc_hist, label=modstring, color=cols[i])
            elapsed, train_loss_hist[end], train_acc_hist[end], test_loss_hist[end], test_acc_hist[end]
        end
        push!(df, (modstring, train_loss, train_acc, test_loss, test_acc), promote=true)
    end
    GC.gc(true)
end
f[1, 2] = Legend(f, ax, "Models", framevisible = false)
display(f)
display(df)
print_timer(to; sortby = :firstexec)
end
run()
