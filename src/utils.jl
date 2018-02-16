const deps = joinpath(@__DIR__, "..", "deps")
const url = "https://github.com/FluxML/Metalhead.jl/releases/download/Models"

function getweights(name)
  mkpath(deps)
  cd(deps) do
    isfile(name) || download("$url/$name", name)
  end
end

function weights(name)
  getweights(name)
  open(deserialize, joinpath(deps, name))
end

function preprocess(im::AbstractMatrix{<:AbstractRGB})
  im = channelview(imresize(im, (224, 224))) .* 255
  im .-= [123.68, 116.779, 103.939]
  im = permutedims(im, (3, 2, 1)) .* trues(1,1,1,1)
end

forward(model, im) = vec(model(preprocess(im)))

topk(classes, ps, k = 5) = sort(collect(zip(classes, ps)), by = x -> -x[2])[1:k]

predict(model, im, k = 5) = topk(imagenet_classes, forward(model, im), k)

classify(model, im) = Flux.argmax(forward(model, im), imagenet_classes)
