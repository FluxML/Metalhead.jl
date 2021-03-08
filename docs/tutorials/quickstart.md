# Quickstart

{cell=quickstart, display=false, output=false, results=false}
```julia
using Flux, Metalhead
```

Using a model from Metalhead is as simple as selecting a model from the table of [available models](#). For example, below we use the ResNet-50 model with pre-trained weights.
{cell=quickstart}
```julia
using Flux, Metalhead

model = resnet50(pretrain=true)
```

Now, we can use this model with Flux like any other model. Below, we train it on some randomly generated data.
{cell=quickstart}
```julia
using Flux: onehotbatch
using Statistics: mean

batchsize = 8
data = [(rand(Float32, 224, 224, 3, batchsize), onehotbatch(rand(1:1000), 1:1000))
        for _ in 1:3]
opt = ADAM()
ps = Flux.params(model)
loss(x, y, m) = Flux.Losses.logitcrossentropy(m(x), y)
cb = () -> @show mean(loss(x, y, model) for (x, y) in data)
Flux.train!((x, y) -> loss(x, y, model), ps, data, opt; cb = cb)
```
