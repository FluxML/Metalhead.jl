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
    cat_channels(x, y)

Concatenate `x` and `y` along the channel dimension (third dimension).
Equivalent to `cat(x, y; dims=3)`.
Convenient binary reduction operator for use with `Parallel`.
"""
cat_channels(x, y) = cat(x, y; dims = 3)

"""
    weights(model)

Load the pre-trained weights for `model` using the stored artifacts.
"""
function weights(model)
  path = joinpath(@artifact_str(model), "$model.bson")

  if ispath(path)
    return BSON.load(path, @__MODULE__)[:weights]
  else
    throw(ArgumentError("No pre-trained weights available for $model."))
  end
end

"""
    loadpretrain!(model, name)

Load the pre-trained weight artifacts matching `<name>.bson` into `model`.
"""
loadpretrain!(model, name) = Flux.loadparams!(model, weights(name))

function _maybe_big_show(io, model)
  if isdefined(Flux, :_big_show)
    if isnothing(get(io, :typeinfo, nothing)) # e.g. top level in REPL
      Flux._big_show(io, model)
    else
      show(io, model)
    end
  else
    show(io, model)
  end
end
