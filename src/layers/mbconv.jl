"""
    dwsep_conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
                    activation = relu; norm_layer = BatchNorm, stride::Integer = 1,
                    bias::Bool = !(norm_layer !== identity), pad::Integer = 0, [bias, weight, init])

Create a depthwise separable convolution chain as used in MobileNetv1.
This is sequence of layers:

  - a `kernel_size` depthwise convolution from `inplanes => inplanes`
  - a (batch) normalisation layer + `activation` (if `norm_layer !== identity`; otherwise
    `activation` is applied to the convolution output)
  - a `kernel_size` convolution from `inplanes => outplanes`
  - a (batch) normalisation layer + `activation` (if `norm_layer !== identity`; otherwise
    `activation` is applied to the convolution output)

See Fig. 3 in [reference](https://arxiv.org/abs/1704.04861v1).

# Arguments

  - `kernel_size`: size of the convolution kernel (tuple)
  - `inplanes`: number of input feature maps
  - `outplanes`: number of output feature maps
  - `activation`: the activation function for the final layer
  - `norm_layer`: the normalisation layer used. Note that using `identity` as the normalisation
    layer will result in no normalisation being applied.
  - `bias`: whether to use bias in the convolution layers.
  - `stride`: stride of the first convolution kernel
  - `pad`: padding of the first convolution kernel
  - `weight`, `init`: initialization for the convolution kernel (see `Flux.Conv`)
"""
function dwsep_conv_norm(kernel_size::Dims{2}, inplanes::Integer, outplanes::Integer,
                         activation = relu; norm_layer = BatchNorm, stride::Integer = 1,
                         bias::Bool = !(norm_layer !== identity), kwargs...)
    return vcat(conv_norm(kernel_size, inplanes, inplanes, activation; norm_layer,
                          stride, bias, groups = inplanes, kwargs...), # depthwise convolution
                conv_norm((1, 1), inplanes, outplanes, activation; norm_layer, bias)) # pointwise convolution
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

!!! warning
    This function does not handle the residual connection by default. The user must add
    this manually to use this block as a standalone. To construct a model, check out the
    builders, which handle the residual connection and other details.

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
                se_activation = activation,
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
                             activation=se_activation, gate_activation = hardσ))
    end
    # project
    append!(layers, conv_norm((1, 1), explanes, outplanes, identity; norm_layer))
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

!!! warning
    This function does not handle the residual connection by default. The user must add
    this manually to use this block as a standalone. To construct a model, check out the
    builders, which handle the residual connection and other details.

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
