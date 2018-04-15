const deps = joinpath(@__DIR__, "..", "deps")
const url = "https://github.com/FluxML/Metalhead.jl/releases/download/Models"

function testimgs end
function valimgs end

function getweights(name)
  mkpath(deps)
  cd(deps) do
    isfile(name) || Base.download("$url/$name", name)
  end
end

function weights(name)
  getweights(name)
  BSON.load(joinpath(deps, name))
end

load_img(im::AbstractMatrix{<:Color}) = im
load_img(str::AbstractString) = load(str)
load_img(val::ValidationImage) = load_img(val.img)

function preprocess(im::AbstractMatrix{<:AbstractRGB})
  im = channelview(imresize(im, (224, 224))) .* 255
  im .-= [123.68, 116.779, 103.939]
  im = permutedims(im, (3, 2, 1)) .* trues(1,1,1,1)
end

preprocess(im) = preprocess(load(im))

forward(model, im) = vec(model(preprocess(RGB.(im))))

topk(ps::AbstractVector, k::Int = 5) = topk(1:length(ps), ps, k)
topk(classes, ps::AbstractVector, k::Int = 5) = sort(collect(zip(classes, ps)), by = x -> -x[2])[1:k]

make_fname(s::AbstractMatrix{<:Color}) = ""
make_fname(s::String) = s
make_fname(im::ValidationImage) = "$(im.set) Validation #$(im.idx)"

ground_truth(m, s::Union{AbstractMatrix, AbstractString}, result) = (nothing, 0.)
function ground_truth(m::ClassificationModel{Class}, im::ValidationImage, result) where {Class}
    if typeof(im.ground_truth) == Class
        (im.ground_truth, result[im.ground_truth.class])
    elseif method_exists(convert, Tuple{Class, typeof(im.ground_truth)})
        (im.ground_truth, result[convert(Class, im.ground_truth).class])
    else
        (im.ground_truth, 0.0)
    end
end

function predict(model::ClassificationModel{Class}, im, k = 5) where {Class}
    fname = make_fname(im)
    img = load_img(im)
    result = forward(model, img)
    PredictionFrame(fname, img,
        Prediction([Class(x)=>y for (x,y) in topk(result, k)]),
        ground_truth(model, im, result)...)
end
classify(model::ClassificationModel, im) = Flux.argmax(forward(model, load_img(im)), labels(model))

function predict(model::ClassificationModel{Class},
        im::Vector{<:Union{AbstractMatrix, String, ValidationImage}}, k=5) where Class
    error("predict is not implicitly vectorized. Use boradcast syntax: predict.(model, imgs)")
end

struct Prediction
    sorted_predictions::Vector{Pair{ObjectClass, Float32}}
end

struct PredictionFrame
    filename::Union{String, Void}
    img
    prediction::Prediction
    # Ground truth, if known
    ground_truth::Union{Void, ObjectClass}
    ground_truth_confidence::Float32
end
