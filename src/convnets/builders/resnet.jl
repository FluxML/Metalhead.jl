"""
    build_resnet(img_dims, stem, get_layers, block_repeats::AbstractVector{<:Integer},
                 connection, classifier_fn)

Creates a generic ResNet-like model.

!!! info
    
    This is a very generic, flexible but low level function that can be used to create any of the ResNet
    variants. For a more user friendly function, see [`Metalhead.resnet`](@ref).

# Arguments

  - `img_dims`: The dimensions of the input image. This is used to determine the number of feature
    maps to be passed to the classifier. This should be a tuple of the form `(height, width, channels)`.
  - `stem`: The stem of the ResNet model. The stem should be created outside of this function and
    passed in as an argument. This is done to allow for more flexibility in creating the stem.
    [`resnet_stem`](@ref) is a helper function that Metalhead provides which is recommended for
    creating the stem.
  - `get_layers` is a function that takes in two inputs - the `stage_idx`, or the index of
    the stage, and the `block_idx`, or the index of the block within the stage. It returns a
    tuple of layers. If the tuple returned by `get_layers` has more than one element, then
    `connection` is used to splat this tuple into `Parallel` - if not, then the only element of
    the tuple is directly inserted into the network. `get_layers` is a very specific function
    and should not be created on its own. Instead, use one of the builders provided by Metalhead
    to create it.
  - `block_repeats`: This is a `Vector` of integers that specifies the number of repeats of each
    block in each stage.
  - `connection`: This is a function that determines the residual connection in the model. For
    `resnets`, either of [`Metalhead.addact`](@ref) or [`Metalhead.actadd`](@ref) is recommended.
  - `classifier_fn`: This is a function that takes in the number of feature maps and returns a
    classifier. This is usually built as a closure using a function like [`Metalhead.create_classifier`](@ref).
    For example, if the number of output classes is `nclasses`, then the function can be defined as
    `channels -> create_classifier(channels, nclasses)`.
"""
function build_resnet(img_dims, stem, get_layers, block_repeats::AbstractVector{<:Integer},
                      connection, classifier_fn)
    # Build stages of the ResNet
    stage_blocks = cnn_stages(get_layers, block_repeats, connection)
    backbone = Chain(stem, stage_blocks...)
    # Add classifier to the backbone
    nfeaturemaps = Flux.outputsize(backbone, img_dims; padbatch = true)[3]
    return Chain(backbone, classifier_fn(nfeaturemaps))
end
