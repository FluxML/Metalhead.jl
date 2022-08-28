"""
    dwsep_conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
                    activation = relu; eps::Float32 = 1.0f-5, revnorm::Bool = false, 
                    stride::Integer = 1, use_norm::NTuple{2, Bool} = (true, true),
                    pad::Integer = 0, [bias, weight, init])

Create a depthwise separable convolution chain as used in MobileNetv1.
This is sequence of layers:

  - a `kernel_size` depthwise convolution from `inplanes => inplanes`
  - a (batch) normalisation layer + `activation` (if `use_norm[1] == true`; otherwise
    `activation` is applied to the convolution output)
  - a `kernel_size` convolution from `inplanes => outplanes`
  - a (batch) normalisation layer + `activation` (if `use_norm[2] == true`; otherwise
    `activation` is applied to the convolution output)

See Fig. 3 in [reference](https://arxiv.org/abs/1704.04861v1).

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `revnorm`: set to `true` to place the batch norm before the convolution
  - `use_norm`: a tuple of two booleans to specify whether to use normalization for the first and
    second convolution
  - `bias`: a tuple of two booleans to specify whether to use bias for the first and second
    convolution. This is set to `(false, false)` by default if `use_norm[0] == true` and
    `use_norm[1] == true`.
  - `stride`: stride of the first convolution kernel
  - `pad`: padding of the first convolution kernel
  - `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](@ref))
"""
function dwsep_conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
                         activation = relu; norm_layer = BatchNorm, eps::Float32 = 1.0f-5,
                         use_norm::NTuple{2, Bool} = (true, true), stride::Integer = 1,
                         bias::NTuple{2, Bool} = (!use_norm[1], !use_norm[2]), kwargs...)
    return vcat(conv_norm(kernel_size, inplanes, inplanes, activation; eps, norm_layer,
                          use_norm = use_norm[1], stride, bias = bias[1],
                          groups = inplanes, kwargs...), # depthwise convolution
                conv_norm((1, 1), inplanes, outplanes, activation; eps, norm_layer,
                          use_norm = use_norm[2], bias = bias[2])) # pointwise convolution
end

"""
    mbconv(kernel_size::Dims{2}, inplanes::Integer, explanes::Integer,
           outplanes::Integer, activation = relu; stride::Integer,
           reduction::Union{Nothing, Real} = nothing,
           se_round_fn = x -> round(Int, x), norm_layer = BatchNorm, kwargs...)

Create a basic inverted residual block for MobileNet and Efficient variants.
This is a sequence of layers:

  - a 1x1 convolution from `inplanes => explanes` followed by a (batch) normalisation layer

  - `activation` if `inplanes != explanes`
  - a `kernel_size` depthwise separable convolution from `explanes => explanes`
  - a (batch) normalisation layer
  - a squeeze-and-excitation block (if `reduction != nothing`) from
    `explanes => se_round_fn(explanes / reduction)` and back to `explanes`
  - a 1x1 convolution from `explanes => outplanes`
  - a (batch) normalisation layer + `activation`

First introduced in the MobileNetv2 paper.
(See Fig. 3 in [reference](https://arxiv.org/abs/1801.04381v4).)

# Arguments

  - `kernel_size`: kernel size of the convolutional layers
  - `inplanes`: number of input feature maps
  - `explanes`: The number of expanded feature maps. This is the number of feature maps
    after the first 1x1 convolution.
  - `outplanes`: The number of output feature maps
  - `activation`: The activation function for the first two convolution layer
  - `stride`: The stride of the convolutional kernel, has to be either 1 or 2
  - `reduction`: The reduction factor for the number of hidden feature maps
    in a squeeze and excite layer (see [`squeeze_excite`](@ref))
  - `se_round_fn`: The function to round the number of reduced feature maps
    in the squeeze and excite layer
  - `norm_layer`: The normalization layer to use
"""
function mbconv(kernel_size::Dims{2}, inplanes::Integer, explanes::Integer,
                outplanes::Integer, activation = relu; stride::Integer,
                reduction::Union{Nothing, Real} = nothing,
                se_round_fn = x -> round(Int, x), norm_layer = BatchNorm)
    @assert stride in [1, 2] "`stride` has to be 1 or 2 for `mbconv`"
    layers = []
    # expand
    if inplanes != explanes
        append!(layers,
                conv_norm((1, 1), inplanes, explanes, activation; norm_layer))
    end
    # depthwise
    append!(layers,
            conv_norm(kernel_size, explanes, explanes, activation; norm_layer,
                      stride, pad = SamePad(), groups = explanes))
    # squeeze-excite layer
    if !isnothing(reduction)
        push!(layers,
              squeeze_excite(explanes; round_fn = se_round_fn, reduction,
                             activation, gate_activation = hardσ))
    end
    # project
    append!(layers, conv_norm((1, 1), explanes, outplanes, identity))
    return Chain(layers...)
end

"""
    fused_mbconv(kernel_size::Dims{2}, inplanes::Integer, explanes::Integer,
                 outplanes::Integer, activation = relu;
                 stride::Integer, norm_layer = BatchNorm)

Create a fused inverted residual block.

This is a sequence of layers:

  - a `kernel_size` depthwise separable convolution from `explanes => explanes`
  - a (batch) normalisation layer
  - a 1x1 convolution from `explanes => outplanes` followed by a (batch) normalisation
    layer + `activation` if `inplanes != explanes`

Originally introduced by Google in [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html).
Later used in the EfficientNetv2 paper.

# Arguments

  - `kernel_size`: kernel size of the convolutional layers
  - `inplanes`: number of input feature maps
  - `explanes`: The number of expanded feature maps
  - `outplanes`: The number of output feature maps
  - `activation`: The activation function for the first two convolution layer
  - `stride`: The stride of the convolutional kernel, has to be either 1 or 2
  - `norm_layer`: The normalization layer to use
"""
function fused_mbconv(kernel_size::Dims{2}, inplanes::Integer,
                      explanes::Integer, outplanes::Integer, activation = relu;
                      stride::Integer, norm_layer = BatchNorm)
    @assert stride in [1, 2] "`stride` has to be 1 or 2 for `fused_mbconv`"
    layers = []
    # fused expand
    explanes = explanes == inplanes ? outplanes : explanes
    append!(layers,
            conv_norm(kernel_size, inplanes, explanes, activation; norm_layer, stride,
                      pad = SamePad()))
    if explanes != inplanes
        # project
        append!(layers, conv_norm((1, 1), explanes, outplanes, identity; norm_layer))
    end
    return Chain(layers...)
end
