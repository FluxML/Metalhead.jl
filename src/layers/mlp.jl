"""
    mlp_block(inplanes::Integer, hidden_planes::Integer, outplanes::Integer = inplanes; 
              dropout_rate = 0., activation = gelu)

Feedforward block used in many MLPMixer-like and vision-transformer models.

# Arguments

  - `inplanes`: Number of dimensions in the input.
  - `hidden_planes`: Number of dimensions in the intermediate layer.
  - `outplanes`: Number of dimensions in the output - by default it is the same as `inplanes`.
  - `dropout_rate`: Dropout rate.
  - `activation`: Activation function to use.
"""
function mlp_block(inplanes::Integer, hidden_planes::Integer, outplanes::Integer = inplanes;
                   dropout_rate = 0.0, activation = gelu)
    return Chain(Dense(inplanes, hidden_planes, activation), Dropout(dropout_rate),
                 Dense(hidden_planes, outplanes), Dropout(dropout_rate))
end

"""
    gated_mlp(gate_layer, inplanes::Integer, hidden_planes::Integer, 
              outplanes::Integer = inplanes; dropout_rate = 0.0, activation = gelu)

Feedforward block based on the implementation in the paper "Pay Attention to MLPs".
([reference](https://arxiv.org/abs/2105.08050))

# Arguments

  - `gate_layer`: Layer to use for the gating.
  - `inplanes`: Number of dimensions in the input.
  - `hidden_planes`: Number of dimensions in the intermediate layer.
  - `outplanes`: Number of dimensions in the output - by default it is the same as `inplanes`.
  - `dropout_rate`: Dropout rate.
  - `activation`: Activation function to use.
"""
function gated_mlp_block(gate_layer, inplanes::Integer, hidden_planes::Integer,
                         outplanes::Integer = inplanes; dropout_rate = 0.0,
                         activation = gelu)
    @assert hidden_planes % 2==0 "`hidden_planes` must be even for gated MLP"
    return Chain(Dense(inplanes, hidden_planes, activation),
                 Dropout(dropout_rate),
                 gate_layer(hidden_planes),
                 Dense(hidden_planes ÷ 2, outplanes),
                 Dropout(dropout_rate))
end
gated_mlp_block(::typeof(identity), args...; kwargs...) = mlp_block(args...; kwargs...)

"""
    create_classifier(inplanes::Integer, nclasses::Integer, activation = identity;
                      pool_layer = AdaptiveMeanPool((1, 1)),
                      dropout_rate = 0.0, use_conv::Bool = false)

Creates a classifier head to be used for models.

# Arguments

  - `inplanes`: number of input feature maps
  - `nclasses`: number of output classes
  - `activation`: activation function to use
  - `pool_layer`: pooling layer to use. This is passed in with the layer instantiated with
    any arguments that are needed i.e. as `AdaptiveMeanPool((1, 1))`, for example.
  - `dropout_rate`: dropout rate used in the classifier head.
  - `use_conv`: whether to use a 1x1 convolutional layer instead of a `Dense` layer.
"""
function create_classifier(inplanes::Integer, nclasses::Integer, activation = identity;
                           use_conv::Bool = false, pool_layer = AdaptiveMeanPool((1, 1)),
                           dropout_rate = nothing)
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
    isnothing(dropout_rate) ? nothing : push!(classifier, Dropout(dropout_rate))
    # Fully-connected layer
    use_conv ? push!(classifier, Conv((1, 1), inplanes => nclasses, activation)) :
    push!(classifier, Dense(inplanes => nclasses, activation))
    return Chain(classifier...)
end
