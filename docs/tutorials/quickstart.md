# Quickstart

{cell=quickstart, display=false, output=false, results=false}
```julia
using Flux, Metalhead
```

Using a model from Metalhead is as simple as selecting a model from the table of [available models](#). For example, below we use the ResNet-18 model.
{cell=quickstart}
```julia
using Flux, Metalhead

model = ResNet(18)
```

Now, we can use this model with Flux like any other model. Below, we train it on some randomly generated data.
{cell=quickstart}
```julia
using Flux: onehotbatch

batchsize = 1
data = [(rand(Float32, 224, 224, 3, batchsize), onehotbatch(rand(1:1000, batchsize), 1:1000))
        for _ in 1:3]
opt = ADAM()
ps = Flux.params(model)
loss(x, y, m) = Flux.Losses.logitcrossentropy(m(x), y)
for (i, (x, y)) in enumerate(data)
    @info "Starting batch $i ..."
    gs = gradient(() -> loss(x, y, model), ps)
    Flux.update!(opt, ps, gs)
end
```
