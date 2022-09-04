"""
    create_classifier(inplanes::Integer, nclasses::Integer, activation = identity;
                      use_conv::Bool = false, pool_layer = AdaptiveMeanPool((1, 1)), 
                      dropout_prob = nothing)

Creates a classifier head to be used for models.

# Arguments

  - `inplanes`: number of input feature maps
  - `nclasses`: number of output classes
  - `activation`: activation function to use
  - `use_conv`: whether to use a 1x1 convolutional layer instead of a `Dense` layer.
  - `pool_layer`: pooling layer to use. This is passed in with the layer instantiated with
    any arguments that are needed i.e. as `AdaptiveMeanPool((1, 1))`, for example.
  - `dropout_prob`: dropout probability used in the classifier head. Set to `nothing` to disable dropout.
"""
function create_classifier(inplanes::Integer, nclasses::Integer, activation = identity;
                           use_conv::Bool = false, pool_layer = AdaptiveMeanPool((1, 1)),
                           dropout_prob = nothing)
    # Decide whether to flatten the input or not
    flatten_in_pool = !use_conv && pool_layer !== identity
    if use_conv
        @assert pool_layer === identity
        "`pool_layer` must be identity if `use_conv` is true"
    end
    classifier = []
    if flatten_in_pool
        push!(classifier, pool_layer, MLUtils.flatten)
    else
        push!(classifier, pool_layer)
    end
    # Dropout is applied after the pooling layer
    isnothing(dropout_prob) ? nothing : push!(classifier, Dropout(dropout_prob))
    # Fully-connected layer
    if use_conv
        push!(classifier, Conv((1, 1), inplanes => nclasses, activation))
    else
        push!(classifier, Dense(inplanes => nclasses, activation))
    end
    return Chain(classifier...)
end

"""
    create_classifier(inplanes::Integer, hidden_planes::Integer, nclasses::Integer,
                      activations::NTuple{2} = (relu, identity);
                      use_conv::NTuple{2, Bool} = (false, false),
                      pool_layer = AdaptiveMeanPool((1, 1)), dropout_prob = nothing)

Creates a classifier head to be used for models with an extra hidden layer.

# Arguments

  - `inplanes`: number of input feature maps
  - `hidden_planes`: number of hidden feature maps
  - `nclasses`: number of output classes
  - `activations`: activation functions to use for the hidden and output layers. This is a
    tuple of two elements, the first being the activation function for the hidden layer and the
    second for the output layer.
  - `use_conv`: whether to use a 1x1 convolutional layer instead of a `Dense` layer. This
    is a tuple of two booleans, the first for the hidden layer and the second for the output
    layer.
  - `pool_layer`: pooling layer to use. This is passed in with the layer instantiated with
    any arguments that are needed i.e. as `AdaptiveMeanPool((1, 1))`, for example.
  - `dropout_prob`: dropout probability used in the classifier head. Set to `nothing` to disable dropout.
"""
function create_classifier(inplanes::Integer, hidden_planes::Integer, nclasses::Integer,
                           activations::NTuple{2, Any} = (relu, identity);
                           use_conv::NTuple{2, Bool} = (false, false),
                           pool_layer = AdaptiveMeanPool((1, 1)), dropout_prob = nothing)
    fc_layers = [uc ? Conv$(1, 1) : Dense for uc in use_conv]
    # Decide whether to flatten the input or not
    flatten_in_pool = !use_conv[1] && pool_layer !== identity
    if use_conv[1]
        @assert pool_layer === identity
        "`pool_layer` must be identity if `use_conv[1]` is true"
    end
    classifier = []
    if flatten_in_pool
        push!(classifier, pool_layer, MLUtils.flatten)
    else
        push!(classifier, pool_layer)
    end
    # first fully-connected layer
    if !isnothing(hidden_planes)
        push!(classifier, fc_layers[1](inplanes => hidden_planes, activations[1]))
    end
    # Dropout is applied after the first dense layer
    isnothing(dropout_prob) ? nothing : push!(classifier, Dropout(dropout_prob))
    # second fully-connected layer
    push!(classifier, fc_layers[2](hidden_planes => nclasses, activations[2]))
    return Chain(classifier...)
end
