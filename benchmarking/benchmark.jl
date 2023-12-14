
using CUDA, cuDNN
using Flux
using Flux: logitcrossentropy, onecold, onehotbatch
using Metalhead
using MLDatasets
using Optimisers
using ProgressMeter
using TimerOutputs
using UnicodePlots

include("tooling.jl")

epochs = 45
batchsize = 1000
device = gpu
allow_skips = true

train_loader, test_loader, labels = load_cifar10(; batchsize)
nlabels = length(labels)
firstbatch = first(first(train_loader))
imsize = size(firstbatch)[1:2]

@info "Benchmarking" epochs batchsize device imsize

to = TimerOutput()

# these should all be the smallest variant of each that is tested in `/test`
modelstrings = (
    "AlexNet()",
    "VGG(11, batchnorm=true)",
    "SqueezeNet()",
    "ResNet(18)",
    "WideResNet(50)",
    "ResNeXt(50, cardinality=32, base_width=4)",
    "SEResNet(18)",
    "SEResNeXt(50, cardinality=32, base_width=4)",
    "Res2Net(50, base_width=26, scale=4)",
    "Res2NeXt(50)",
    "GoogLeNet(batchnorm=true)",
    "DenseNet(121)",
    "Inceptionv3()",
    "Inceptionv4()",
    "InceptionResNetv2()",
    "Xception()",
    "MobileNetv1(0.5)",
    "MobileNetv2(0.5)",
    "MobileNetv3(:small, width_mult=0.5)",
    "MNASNet(:A1, width_mult=0.5)",
    "EfficientNet(:b0)",
    "EfficientNetv2(:small)",
    "ConvMixer(:small)",
    "ConvNeXt(:small)",
    # "MLPMixer()", # no tests found
    # "ResMLP()", # no tests found
    # "gMLP()", # no tests found
    "ViT(:tiny)",
    "UNet()"
    )

for (i, modstring) in enumerate(modelstrings)
    @timeit to "$modstring" begin
        @info "Evaluating $i/$(length(modelstrings)): $modstring"
        # Initial precompile is variable based on what came before, so don't time first load
        eval(Meta.parse(modstring))
        # second load simulates what might be possible with a proper set-up pkgimage workload
        @timeit to "Load" model=eval(Meta.parse(modstring))
        @timeit to "Training" train(model,
            train_loader,
            test_loader;
            limit = 1,
            to,
            device)||(allow_skips || break)
    end
end
print_timer(to; sortby = :firstexec)
