"""
    conv_bn(kernelsize, inplanes, outplanes, activation = relu;
            rev = false, preact = true,
            stride = 1, pad = 0, dilation = 1, groups = 1, [bias, weight, init],
            initβ = Flux.zeros32, initγ = Flux.ones32, ϵ = 1f-5, momentum = 1f-1)

Create a convolution + batch normalization pair with ReLU activation.

# Arguments
- `kernelsize`: size of the convolution kernel (tuple)
- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `activation`: the activation function for the final layer
- `rev`: set to `true` to place the batch norm before the convolution
- `preact`: set to `true` to place the activation function before the batch norm
            (only compatible with `rev = false`)
- `stride`: stride of the convolution kernel
- `pad`: padding of the convolution kernel
- `dilation`: dilation of the convolution kernel
- `groups`: groups for the convolution kernel
- `bias`, `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](#))
- `initβ`, `initγ`: initialization for the batch norm (see [`Flux.BatchNorm`](#))
- `ϵ`, `momentum`: batch norm parameters (see [`Flux.BatchNorm`](#))
"""
function conv_bn(kernelsize, inplanes, outplanes, activation = relu;
                 rev = false, preact = false,
                 initβ = Flux.zeros32, initγ = Flux.ones32, ϵ = 1f-5, momentum = 1f-1,
                 kwargs...)
  layers = []

  if rev
    activations = (conv = activation, bn = identity)
    bnplanes = inplanes
  else
    activations = (conv = identity, bn = activation)
    bnplanes = outplanes
  end

  if preact
    rev ? throw(ArgumentError("preact and rev cannot be set at the same time")) :
          activations = (conv = activation, bn = identity)
  end

  push!(layers, Conv(kernelsize, Int(inplanes) => Int(outplanes), activations.conv; kwargs...))
  push!(layers, BatchNorm(Int(bnplanes), activations.bn;
                          initβ = initβ, initγ = initγ, ϵ = ϵ, momentum = momentum))

  return rev ? reverse(layers) : layers
end


"""
    skip_projection(inplanes, outplanes, downsample = false)

Create a skip projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: set to `true` to downsample the input
"""
skip_projection(inplanes, outplanes, downsample = false) = downsample ? 
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 2, bias = false)...) :
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 1, bias = false)...)

# array -> PaddedView(0, array, outplanes) for zero padding arrays
"""
    skip_identity(inplanes, outplanes[, downsample])

Create a identity projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: this argument is ignored but it is needed for compatibility with [`resnet`](#).
"""
function skip_identity(inplanes, outplanes)
  if outplanes > inplanes
    return Chain(MaxPool((1, 1), stride = 2),
                 y -> cat(y, zeros(eltype(y),
                                   size(y, 1),
                                   size(y, 2),
                                   outplanes - inplanes, size(y, 4)); dims = 3))
  else
    return identity
  end
end
skip_identity(inplanes, outplanes, downsample) = skip_identity(inplanes, outplanes)

"""
    squeeze_excite(channels, reduction = 4)

Squeeze and excitation layer used by MobileNet variants
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `channels`: the number of input/output feature maps
- `reduction = 4`: the reduction factor for the number of hidden feature maps
                   (must be >= 1)
"""
function squeeze_excite(channels, reduction = 4)
  @assert (reduction >= 1) "`reduction` must be >= 1"
  SkipConnection(Chain(AdaptiveMeanPool((1, 1)),
                       conv_bn((1, 1), channels, channels ÷ reduction, relu; bias = false)...,
                       conv_bn((1, 1), channels ÷ reduction, channels, hardσ)...), .*)
end

"""
    invertedresidual(kernel_size, inplanes, hidden_planes, outplanes, activation = relu;
                     stride, reduction = nothing)

Create a basic inverted residual block for MobileNet variants
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments
- `kernel_size`: The kernel size of the convolutional layers
- `inplanes`: The number of input feature maps
- `hidden_planes`: The number of feature maps in the hidden layer
- `outplanes`: The number of output feature maps
- `activation`: The activation function for the first two convolution layer
- `stride`: The stride of the convolutional kernel, has to be either 1 or 2
- `reduction`: The reduction factor for the number of hidden feature maps
               in a squeeze and excite layer (see [`squeeze_excite`](#)).
               Must be >= 1 or `nothing` for no squeeze and excite layer.
"""
function invertedresidual(kernel_size, inplanes, hidden_planes, outplanes, activation = relu;
                          stride, reduction = nothing)
  @assert stride in [1, 2] "`stride` has to be 1 or 2"

  pad = @. (kernel_size - 1) ÷ 2
  conv1 = (inplanes == hidden_planes) ? () : conv_bn((1, 1), inplanes, hidden_planes, activation; bias = false)
  selayer = isnothing(reduction) ? identity : squeeze_excite(hidden_planes, reduction)

  invres = Chain(conv1...,
                 conv_bn(kernel_size, hidden_planes, hidden_planes, activation;
                         bias = false, stride, pad = pad, groups = hidden_planes)...,
                 selayer,
                 conv_bn((1, 1), hidden_planes, outplanes, identity; bias = false)...)

  (stride == 1 && inplanes == outplanes) ? SkipConnection(invres, +) : invres
end
invertedresidual(kernel_size::Integer, args...; kwargs...) =
  invertedresidual((kernel_size, kernel_size), args...; kwargs...)
