"""
    conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
              activation = relu; norm_layer = BatchNorm, revnorm::Bool = false,
              preact::Bool = false, stride::Integer = 1, pad::Integer = 0,
              dilation::Integer = 1, groups::Integer = 1, [bias, weight, init])

Create a convolution + normalisation layer pair with activation.

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `norm_layer`: the normalisation layer used. Note that using `identity` as the normalisation
    layer will result in no normalisation being applied. (This is only compatible with `preact`
    and `revnorm` both set to `false`.)
  - `revnorm`: set to `true` to place the normalisation layer before the convolution
  - `preact`: set to `true` to place the activation function before the normalisation layer
    (only compatible with `revnorm = false`)
  - `bias`: bias for the convolution kernel. This is set to `false` by default if
    `norm_layer` is not `identity` and `true` otherwise.
  - `stride`: stride of the convolution kernel
  - `pad`: padding of the convolution kernel
  - `dilation`: dilation of the convolution kernel
  - `groups`: groups for the convolution kernel
  - `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](@ref))
"""
function conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
                   activation = relu; norm_layer = BatchNorm, revnorm::Bool = false,
                   preact::Bool = false, bias = !(norm_layer !== identity), kwargs...)
    # no normalization layer
    if !(norm_layer !== identity)
        if preact || revnorm
            throw(ArgumentError("`preact` only supported with `norm_layer !== identity`.
            Check if a non-`identity` norm layer is intended."))
        else
            # early return if no norm layer is required
            return [Conv(kernel_size, inplanes => outplanes, activation; kwargs...)]
        end
    end
    # channels for norm layer and activation functions for both conv and norm
    if revnorm
        activations = (conv = activation, norm = identity)
        normplanes = inplanes
    else
        activations = (conv = identity, norm = activation)
        normplanes = outplanes
    end
    # handle pre-activation
    if preact
        if revnorm
            throw(ArgumentError("`preact` and `revnorm` cannot be set at the same time"))
        else
            activations = (conv = activation, norm = identity)
        end
    end
    # layers
    layers = [Conv(kernel_size, inplanes => outplanes, activations.conv; bias, kwargs...),
        norm_layer(normplanes, activations.norm)]
    return revnorm ? reverse(layers) : layers
end

"""
    basic_conv_bn(kernel_size::Dims{2}, inplanes, outplanes, activation = relu;
                  kwargs...)

Returns a convolution + batch normalisation pair with activation as used by the
Inception family of models with default values matching those used in the official
TensorFlow implementation.

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `batchnorm`: set to `true` to include batch normalization after each convolution
  - `kwargs`: keyword arguments passed to [`conv_norm`](@ref)
"""
function basic_conv_bn(kernel_size::Dims{2}, inplanes, outplanes, activation = relu;
                       batchnorm::Bool = true, kwargs...)
    # TensorFlow uses a default epsilon of 1e-3 for BatchNorm
    norm_layer = batchnorm ?
                 (args...; kwargs...) -> BatchNorm(args...; ϵ = 1.0f-3, kwargs...) :
                 identity
    return conv_norm(kernel_size, inplanes, outplanes, activation; norm_layer, kwargs...)
end
