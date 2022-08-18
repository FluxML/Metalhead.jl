"""
    conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
              activation = relu; norm_layer = BatchNorm, revnorm::Bool = false,
              eps::Float32 = 1.0f-5, preact::Bool = false, use_norm::Bool = true,
              stride::Integer = 1, pad::Integer = 0, dilation::Integer = 1, 
              groups::Integer = 1, [bias, weight, init])

    conv_norm(kernel_size::Dims{2}, inplanes => outplanes, activation = identity;
              kwargs...)

Create a convolution + batch normalization pair with activation.

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `norm_layer`: the normalization layer used
  - `revnorm`: set to `true` to place the normalisation layer before the convolution
  - `preact`: set to `true` to place the activation function before the batch norm
    (only compatible with `revnorm = false`)
  - `use_norm`: set to `false` to disable normalization
    (only compatible with `revnorm = false` and `preact = false`)
  - `stride`: stride of the convolution kernel
  - `pad`: padding of the convolution kernel
  - `dilation`: dilation of the convolution kernel
  - `groups`: groups for the convolution kernel
  - `bias`: bias for the convolution kernel. This is set to `false` by default if
    `use_norm = true`.
  - `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](#))
"""
function conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
                   activation = relu; norm_layer = BatchNorm, revnorm::Bool = false,
                   eps::Float32 = 1.0f-5, preact::Bool = false, use_norm::Bool = true,
                   bias = !use_norm, kwargs...)
    # no normalization layer
    if !use_norm
        if preact || revnorm
            throw(ArgumentError("`preact` only supported with `use_norm = true`"))
        else
            # early return if no norm layer is required
            return [Conv(kernel_size, inplanes => outplanes, activation; kwargs...)]
        end
    end
    # channels for norm layer and activation functions for both conv and norm
    if revnorm
        activations = (conv = activation, bn = identity)
        normplanes = inplanes
    else
        activations = (conv = identity, bn = activation)
        normplanes = outplanes
    end
    # handle pre-activation
    if preact
        if revnorm
            throw(ArgumentError("`preact` and `revnorm` cannot be set at the same time"))
        else
            activations = (conv = activation, bn = identity)
        end
    end
    # layers
    layers = [Conv(kernel_size, inplanes => outplanes, activations.conv; bias, kwargs...),
        norm_layer(normplanes, activations.bn; ϵ = eps)]
    return revnorm ? reverse(layers) : layers
end
function conv_norm(kernel_size::Dims{2}, ch::Pair{<:Integer, <:Integer},
                   activation = identity; kwargs...)
    inplanes, outplanes = ch
    return conv_norm(kernel_size, inplanes, outplanes, activation; kwargs...)
end

# conv + bn layer combination as used by the inception model family matching
# the default values used in TensorFlow
function basic_conv_bn(kernel_size::Dims{2}, inplanes, outplanes, activation = relu;
                       kwargs...)
    return conv_norm(kernel_size, inplanes, outplanes, activation; norm_layer = BatchNorm,
                     eps = 1.0f-3, kwargs...)
end
