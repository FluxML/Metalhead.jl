"""
    conv_bn(kernelsize, inplanes, outplanes;
            stride = 1, pad = 0, usebias = true, rev = false)

Create a convolution + batch normalization pair with ReLU activation.

# Arguments
- `kernelsize`: size of the convolution kernel (tuple)
- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `stride`: stride of the convolution kernel
- `pad`: padding of the convolution kernel
- `usebias`: set to `true` to use a bias in the convolution layer
- `rev`: set to `true` to place the batch norm before the convolution
"""
function conv_bn(kernelsize, inplanes, outplanes;
                 stride = 1, pad = 0, usebias = true, rev = false)
  conv_layer = []
  if usebias
    push!(conv_layer, Conv(kernelsize, inplanes => outplanes,
                           stride = stride, pad = pad, init = Flux.kaiming_normal))
  else
    push!(conv_layer, Conv(kernelsize, inplanes => outplanes,
                           stride = stride,
                           pad = pad,
                           init = Flux.kaiming_normal,
                           bias = Flux.Zeros()))
  end

  if rev
    push!(conv_layer, BatchNorm(inplanes, relu))
    return reverse(Tuple(conv_layer))
  end

  push!(conv_layer, BatchNorm(outplanes, relu))
  return Tuple(conv_layer)
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
weights(model) = BSON.load(joinpath(@artifact_str(model), "$model.bson"), @__MODULE__)[:weights]
pretrain_error(model) = throw(ArgumentError("No pre-trained weights available for $model."))
