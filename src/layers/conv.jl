"""
    conv_norm(kernel_size, inplanes::Int, outplanes::Int, activation = relu;
              norm_layer = BatchNorm, prenorm = false, preact = false, use_bn = true,
              stride = 1, pad = 0, dilation = 1, groups = 1, [bias, weight, init])

Create a convolution + batch normalization pair with activation.

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `norm_layer`: the normalization layer used
  - `prenorm`: set to `true` to place the batch norm before the convolution
  - `preact`: set to `true` to place the activation function before the batch norm
    (only compatible with `prenorm = false`)
  - `use_bn`: set to `false` to disable batch normalization
    (only compatible with `prenorm = false` and `preact = false`)
  - `stride`: stride of the convolution kernel
  - `pad`: padding of the convolution kernel
  - `dilation`: dilation of the convolution kernel
  - `groups`: groups for the convolution kernel
  - `bias`, `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](#))
"""
function conv_norm(kernel_size, inplanes::Int, outplanes::Int, activation = relu;
                   norm_layer = BatchNorm, prenorm = false, preact = false, use_bn = true,
                   kwargs...)
    if !use_bn
        if (preact || prenorm)
            throw(ArgumentError("`preact` only supported with `use_bn = true`"))
        else
            return [Conv(kernel_size, inplanes => outplanes, activation; kwargs...)]
        end
    end
    if prenorm
        activations = (conv = activation, bn = identity)
        bnplanes = inplanes
    else
        activations = (conv = identity, bn = activation)
        bnplanes = outplanes
    end
    if preact
        if prenorm
            throw(ArgumentError("`preact` and `prenorm` cannot be set at the same time"))
        else
            activations = (conv = activation, bn = identity)
        end
    end
    layers = [Conv(kernel_size, inplanes => outplanes, activations.conv; kwargs...),
        norm_layer(bnplanes, activations.bn)]
    return prenorm ? reverse(layers) : layers
end

function conv_norm(kernel_size, ch::Pair{<:Integer, <:Integer}, outplanes,
                   activation = identity; kwargs...)
    inplanes, outplanes = ch
    return conv_norm(kernel_size, inplanes, outplanes, activation; kwargs...)
end

"""
    depthwise_sep_conv_bn(kernel_size, inplanes, outplanes, activation = relu;
                               prenorm = false, use_bn = (true, true),
                               stride = 1, pad = 0, dilation = 1, [bias, weight, init])

Create a depthwise separable convolution chain as used in MobileNetv1.
This is sequence of layers:

  - a `kernel_size` depthwise convolution from `inplanes => inplanes`
  - a batch norm layer + `activation` (if `use_bn[1] == true`; otherwise `activation` is applied to the convolution output)
  - a `kernel_size` convolution from `inplanes => outplanes`
  - a batch norm layer + `activation` (if `use_bn[2] == true`; otherwise `activation` is applied to the convolution output)

See Fig. 3 in [reference](https://arxiv.org/abs/1704.04861v1).

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `prenorm`: set to `true` to place the batch norm before the convolution
  - `use_bn`: a tuple of two booleans to specify whether to use batch normalization for the first and second convolution
  - `stride`: stride of the first convolution kernel
  - `pad`: padding of the first convolution kernel
  - `dilation`: dilation of the first convolution kernel
  - `bias`, `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](#))
"""
function depthwise_sep_conv_bn(kernel_size, inplanes, outplanes, activation = relu;
                               prenorm = false, use_bn = (true, true),
                               stride = 1, kwargs...)
    return vcat(conv_norm(kernel_size, inplanes, inplanes, activation;
                          prenorm, use_bn = use_bn[1], stride, groups = inplanes,
                          kwargs...),
                conv_norm((1, 1), inplanes, outplanes, activation; prenorm,
                          use_bn = use_bn[2]))
end

"""
    invertedresidual(kernel_size, inplanes, hidden_planes, outplanes, activation = relu;
                     stride, reduction = nothing)

Create a basic inverted residual block for MobileNet variants
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments

  - `kernel_size`: kernel size of the convolutional layers
  - `inplanes`: number of input feature maps
  - `hidden_planes`: The number of feature maps in the hidden layer
  - `outplanes`: The number of output feature maps
  - `activation`: The activation function for the first two convolution layer
  - `stride`: The stride of the convolutional kernel, has to be either 1 or 2
  - `reduction`: The reduction factor for the number of hidden feature maps
    in a squeeze and excite layer (see [`squeeze_excite`](#)).
"""
function invertedresidual(kernel_size, inplanes, hidden_planes, outplanes,
                          activation = relu; stride, reduction = nothing)
    @assert stride in [1, 2] "`stride` has to be 1 or 2"
    pad = @. (kernel_size - 1) ÷ 2
    conv1 = (inplanes == hidden_planes) ? identity :
            Chain(conv_norm((1, 1), inplanes, hidden_planes, activation; bias = false))
    selayer = isnothing(reduction) ? identity :
              squeeze_excite(hidden_planes; reduction, activation, gate_activation = hardσ,
                             norm_layer = BatchNorm)
    invres = Chain(conv1,
                   conv_norm(kernel_size, hidden_planes, hidden_planes, activation;
                             bias = false, stride, pad = pad, groups = hidden_planes)...,
                   selayer,
                   conv_norm((1, 1), hidden_planes, outplanes, identity; bias = false)...)
    return (stride == 1 && inplanes == outplanes) ? SkipConnection(invres, +) : invres
end

function invertedresidual(kernel_size::Integer, args...; kwargs...)
    return invertedresidual((kernel_size, kernel_size), args...; kwargs...)
end
