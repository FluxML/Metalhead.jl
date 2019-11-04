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

load_img(im::AbstractMatrix{<:Color}) = im
load_img(str::AbstractString) = load(str)
load_img(val::ValidationImage) = load_img(val.img)

# Resize an image such that its smallest dimension is the given length
function resize_smallest_dimension(im, len)
  reduction_factor = len/minimum(size(im)[1:2])
  new_size = size(im)
  new_size = (
      round(Int, size(im,1)*reduction_factor),
      round(Int, size(im,2)*reduction_factor),
  )
  if reduction_factor < 1.0
    # Images.jl's imresize() needs to first lowpass the image, it won't do it for us
    im = imfilter(im, KernelFactors.gaussian(0.75/reduction_factor), Inner())
  end
  return imresize(im, new_size)
end

# Take the len-by-len square of pixels at the center of image `im`
function center_crop(im, len)
  l2 = div(len,2)
  adjust = len % 2 == 0 ? 1 : 0
  return im[div(end,2)-l2:div(end,2)+l2-adjust,div(end,2)-l2:div(end,2)+l2-adjust]
end

function preprocess(im::AbstractMatrix{<:AbstractRGB})
  # Resize such that smallest edge is 256 pixels long
  im = resize_smallest_dimension(im, 256)

  # Center-crop to 224x224
  im = center_crop(im, 224)

  # Convert to channel view and normalize (these coefficients taken
  # from PyTorch's ImageNet normalization code)
  μ = [0.485, 0.456, 0.406]
  σ = [0.229, 0.224, 0.225]
  im = (channelview(im) .- μ)./σ

  # Convert from CHW (Image.jl's channel ordering) to WHCN (Flux.jl's ordering)
  # and enforce Float32, as that seems important to Flux
  return Float32.(permutedims(im, (3, 2, 1))[:,:,:,:].*255)
end

preprocess(im) = preprocess(load(im))
preprocess(im::AbstractMatrix) = preprocess(RGB.(im))

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
