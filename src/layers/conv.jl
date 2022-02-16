"""
    conv_bn(kernelsize, inplanes, outplanes, activation = relu;
            rev = false,
            stride = 1, pad = 0, dilation = 1, groups = 1, [bias, weight, init],
            initβ = Flux.zeros32, initγ = Flux.ones32, ϵ = 1f-5, momentum = 1f-1)

Create a convolution + batch normalization pair with ReLU activation.

# Arguments
- `kernelsize`: size of the convolution kernel (tuple)
- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `activation`: the activation function for the final layer
- `rev`: set to `true` to place the batch norm before the convolution
- `stride`: stride of the convolution kernel
- `pad`: padding of the convolution kernel
- `dilation`: dilation of the convolution kernel
- `groups`: groups for the convolution kernel
- `bias`, `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](#))
- `initβ`, `initγ`: initialization for the batch norm (see [`Flux.BatchNorm`](#))
- `ϵ`, `momentum`: batch norm parameters (see [`Flux.BatchNorm`](#))
"""
function conv_bn(kernelsize, inplanes, outplanes, activation = relu;
                 rev = false,
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