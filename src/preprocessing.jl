## This file defines image preprocessing chains for Flux models, making it easy
#  to do things like train models on ImageNet.  For an example of how to use
#  these preprocessing functions, check out `training/ImageNet/train_imagenet.jl`

"""
    load_img(filename::String)

Thin wrapper around `Images.load()` that immediately converts the resultant
array to a homogenous Float32 tensor.
"""
function load_img(filename::AbstractString)
    # Load the image
    im = load(filename)

    # Permute dimensions to get (R, G, B), then expand to four dimensions to
    # get a singleton batch axis, as the rest of Metalhead expects.
    im = permutedims(channelview(RGB.(im)), (3, 2, 1))[:,:,:,:]

    # Return this as a Float32, (This should no longer be necessary once Flux
    # does the conversion for us, but until then we'll frontload it.)
    return Float32.(im)
end

# If we try to load something that's already loaded, just return it.
load_img(val::ValidationImage) = load_img(val.img)
load_img(im::AbstractArray{T, 4}) where {T} = im


# Resize an image such that its smallest dimension is the given length
function resize_smallest_dimension(im::AbstractArray{T, 4}, len) where {T}
    # Images.jl doesn't like our batch axis, so drop that temporarily
    im = im[:,:,:,1]

    reduction_factor = len/minimum(size(im)[1:2])
    new_size = size(im)
    new_size = (
        round(Int, size(im,1)*reduction_factor),
        round(Int, size(im,2)*reduction_factor),
        new_size[3], # number of channels
    )
    if reduction_factor < 1.0
        # Use restrict() to quarter our size each step, which is much faster
        # than a single large Gaussian imfilter().
        while reduction_factor < 0.5
            im = cat((restrict(im[:,:,cidx]) for cidx in 1:size(im, 3))..., dims=3)
            reduction_factor *= 2
        end
        # low-pass filter
        im = imfilter(im, KernelFactors.gaussian(0.75/reduction_factor), Inner())
    end

    # Expand the result back up to a 4d tensor
    return imresize(im, new_size)[:,:,:,:]
end


"""
    center_crop(im, len)

Extracts the `len`-by-`len` square of pixels centered within `im`.
"""
function center_crop(im::AbstractArray{T, 4}, len::Integer) where {T}
    l2 = div(len,2)
    adjust = len % 2 == 0 ? 1 : 0
    return im[
        div(end,2)-l2 : div(end,2)+l2 - adjust,
        div(end,2)-l2 : div(end,2)+l2 - adjust,
        :, # across all channels
        :, # across all batches
    ]
end


"""
    channel_normalize(im)

Normalizes the channels of `im` according to the standard ImageNet training
coefficiients, yielding roughly unit normal distribution outputs across the
ImageNet corpus.  (These values gratefully taken from PyTorch)
"""
function channel_normalize(im::AbstractArray{T, 4}) where {T}
    # Convert our channel normalization arrays (in R, G, B) order
    # to 1x1x3x1 tensors so that we can use dot-operators to directly
    # subtract and divide to normalize.
    μ = reshape([0.485, 0.456, 0.406], (1, 1, 3, 1))
    σ = reshape([0.229, 0.224, 0.225], (1, 1, 3, 1))
    return (im .- μ)./σ
end



"""
    imagenet_val_preprocess(im)

Perform the typical ImageNet preprocessing steps for validation of a resize,
center crop, and normalization.
"""
function imagenet_val_preprocess(im)
    # Make sure that `im` is loaded
    t_0 = time()
    im = load_img(im)
    t_1 = time()

    # Resize such that smallest edge is 256 pixels long, center-crop to
    # 224x224, then normalize channels and return
    im = resize_smallest_dimension(im, 256)
    t_2 = time()
    im = center_crop(im, 224)
    t_3 = time()
    return (channel_normalize(im), t_1 - t_0, t_2 - t_1, t_3 - t_2)
end


"""
    imagenet_train_preprocess(im)

Perform the typical ImageNet preprocessing steps for training of a random crop,
resize, random flip, and normalization.
"""
function imagenet_train_preprocess(im)
    # TODO: random crop
    return imagenet_val_preprocess(im)
end
