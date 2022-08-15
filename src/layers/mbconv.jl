# TODO add support for stochastic depth to mbconv and fused_mbconv
"""
    mbconv(kernel_size, inplanes::Integer, explanes::Integer,
                     outplanes::Integer, activation = relu; stride::Integer,
                     reduction::Union{Nothing, Integer} = nothing)

Create a basic inverted residual block for MobileNet variants
([reference](https://arxiv.org/abs/1905.02244)).

# Arguments

  - `kernel_size`: kernel size of the convolutional layers
  - `inplanes`: number of input feature maps
  - `explanes`: The number of feature maps in the hidden layer
  - `outplanes`: The number of output feature maps
  - `activation`: The activation function for the first two convolution layer
  - `stride`: The stride of the convolutional kernel, has to be either 1 or 2
  - `reduction`: The reduction factor for the number of hidden feature maps
    in a squeeze and excite layer (see [`squeeze_excite`](#))
"""
function mbconv(kernel_size::Dims{2}, inplanes::Integer, explanes::Integer,
                outplanes::Integer, activation = relu; stride::Integer,
                dilation::Integer = 1, reduction::Union{Nothing, Integer} = nothing,
                norm_layer = BatchNorm)
    @assert stride in [1, 2] "`stride` has to be 1 or 2 for `mbconv`"
    layers = []
    # expand
    if inplanes != explanes
        append!(layers,
                conv_norm((1, 1), inplanes, explanes, activation; norm_layer))
    end
    # depthwise
    stride = dilation > 1 ? 1 : stride
    append!(layers,
            conv_norm(kernel_size, explanes, explanes, activation; norm_layer,
                      stride, dilation, pad = SamePad(), groups = explanes))
    # squeeze-excite layer
    if !isnothing(reduction)
        push!(layers,
              squeeze_excite(explanes, max(1, inplanes ÷ reduction); activation,
                             gate_activation = hardσ))
    end
    # project
    append!(layers, conv_norm((1, 1), explanes, outplanes, identity))
    return Chain(layers...)
end

function fused_mbconv(kernel_size::Dims{2}, inplanes::Integer,
                      explanes::Integer, outplanes::Integer, activation = relu;
                      stride::Integer, norm_layer = BatchNorm)
    @assert stride in [1, 2] "`stride` has to be 1 or 2 for `fused_mbconv`"
    layers = []
    if explanes != inplanes
        # fused expand
        append!(layers,
                conv_norm(kernel_size, inplanes, explanes, activation; norm_layer, stride,
                          pad = SamePad()))
        # project
        append!(layers, conv_norm((1, 1), explanes, outplanes, identity; norm_layer))
    else
        append!(layers,
                conv_norm((1, 1), inplanes, outplanes, activation; norm_layer, stride))
    end
    return Chain(layers...)
end
