const deps = joinpath(@__DIR__, "..", "deps")
const url = "https://github.com/FluxML/Metalhead.jl/releases/download/Models"

function testimgs end
function valimgs end

function getweights(name)
    mkpath(deps)
    cd(deps) do
        if name == "vgg19.bson"
            isfile(name) || Base.download("$url/$name", name)
        else
            loc = "https://github.com/FluxML/Metalhead.jl/releases/download/v0.1.1"
            isfile(name) || Base.download("$loc/$name", name)
        end
    end
end

function weights(name)
    getweights(name)
    BSON.load(joinpath(deps, name))
end

# TODO: Remove after NNlib supports flip kernel through https://github.com/FluxML/NNlib.jl/pull/53
flipkernel(x::AbstractArray) = x[end:-1:1, end:-1:1, :, :]

forward(model, im) = vec(model(imagenet_test_preprocess(im)))

topk(ps::AbstractVector, k::Int = 5) = topk(1:length(ps), ps, k)
topk(classes, ps::AbstractVector, k::Int = 5) = sort(collect(zip(classes, ps)), by = x -> -x[2])[1:k]

make_fname(s::AbstractMatrix{<:Color}) = ""
make_fname(s::String) = s
make_fname(im::ValidationImage) = "$(im.set) Validation #$(im.idx)"

ground_truth(m, s::Union{AbstractMatrix, AbstractString}, result) = (nothing, 0.)
function ground_truth(m::ClassificationModel{Class}, im::ValidationImage, result) where {Class}
    if typeof(im.ground_truth) == Class
        (im.ground_truth, result[im.ground_truth.class])
    elseif hasmethod(convert, Tuple{Class, typeof(im.ground_truth)})
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
classify(model::ClassificationModel, im) = Flux.onecold(forward(model, load_img(im)), labels(model))

function predict(model::ClassificationModel{Class},
        im::Vector{<:Union{AbstractMatrix, String, ValidationImage}}, k=5) where Class
    error("predict is not implicitly vectorized. Use broadcast syntax: predict.(model, imgs)")
end

struct Prediction
    sorted_predictions::Vector{Pair{ObjectClass, Float32}}
end

struct PredictionFrame
    filename::Union{String, Nothing}
    img
    prediction::Prediction
    # Ground truth, if known
    ground_truth::Union{Nothing, ObjectClass}
    ground_truth_confidence::Float32
end
