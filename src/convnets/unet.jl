function pixel_shuffle_icnr(inplanes, outplanes; r = 2)
    return Chain(Chain(basic_conv_bn((1, 1), inplanes, outplanes * (r^2)...)),
                 Flux.PixelShuffle(r))
end

function unet_combine_layer(inplanes, outplanes)
    return Chain(Chain(basic_conv_bn((3, 3), inplanes, outplanes; pad = 1)...),
                 Chain(basic_conv_bn((3, 3), outplanes, outplanes; pad = 1)...))
end

function unet_middle_block(inplanes)
    return Chain(Chain(basic_conv_bn((3, 3), inplanes, 2 * inplanes; pad = 1)...),
                 Chain(basic_conv_bn((3, 3), 2 * inplanes, inplanes; pad = 1)...))
end

function unet_final_block(inplanes, outplanes)
    return Chain(basicblock(inplanes, inplanes; reduction_factor = 1),
                 Chain(basic_conv_bn((1, 1), inplanes, outplanes)...))
end

function unet_block(m_child, inplanes, midplanes, outplanes = 2 * inplanes)
    return Chain(SkipConnection(Chain(m_child,
                                      pixel_shuffle_icnr(midplanes, midplanes)),
                                Parallel(cat_channels, identity, BatchNorm(inplanes))),
                 relu,
                 unet_combine_layer(inplanes + midplanes, outplanes))
end

function unetlayers(layers, sz; outplanes = nothing, skip_upscale = 0,
                    m_middle = _ -> (identity,))
    isempty(layers) && return m_middle(sz[end - 1])

    layer, layers = layers[1], layers[2:end]
    outsz = Flux.outputsize(layer, sz)
    does_downscale = sz[1] ÷ 2 == outsz[1]

    if !does_downscale
        return Chain(layer, unetlayers(layers, outsz; outplanes, skip_upscale)...)
    elseif does_downscale && skip_upscale > 0
        return Chain(layer,
                     unetlayers(layers, outsz; skip_upscale = skip_upscale - 1,
                                outplanes)...)
    else
        childunet = Chain(unetlayers(layers, outsz; skip_upscale)...)
        outsz = Flux.outputsize(childunet, outsz)

        inplanes = sz[end - 1]
        midplanes = outsz[end - 1]
        outplanes = isnothing(outplanes) ? inplanes : outplanes

        return unet_block(Chain(layer, childunet),
                          inplanes, midplanes, outplanes)
    end
end

"""
    unet(encoder_backbone, imgdims, outplanes::Integer, final::Any = unet_final_block,
         fdownscale::Integer = 0)

Creates a UNet model with specified convolutional backbone. 
Backbone of any Metalhead ResNet-like model can be used as encoder 
([reference](https://arxiv.org/abs/1505.04597)).

# Arguments

    - `encoder_backbone`: The backbone layers of specified model to be used as encoder.
    	For example, `Metalhead.backbone(Metalhead.ResNet(18))` can be passed 
    	to instantiate a UNet with layers of resnet18 as encoder.
    - `inputsize`: size of input image
    - `outplanes`: number of output feature planes
    - `final`: final block as described in original paper
    - `fdownscale`: downscale factor
"""
function unet(encoder_backbone, imgdims, inchannels::Integer,outplanes::Integer,
    final::Any = unet_final_block, fdownscale::Integer = 0)
backbonelayers = collect(flatten_chains(encoder_backbone))

# Adjusting input size to include channels
adjusted_imgdims = (imgdims..., inchannels, 1)

layers = unetlayers(backbonelayers, adjusted_imgdims; m_middle = unet_middle_block,
              skip_upscale = fdownscale)

outsz = Flux.outputsize(layers, adjusted_imgdims)
layers = Chain(layers, final(outsz[end - 1], outplanes))

return layers
end
function modify_first_conv_layer(encoder_backbone, inchannels)
    for (index, layer) in enumerate(encoder_backbone.layers)
        if isa(layer, Flux.Conv)  
            # Correctly infer the number of output channels from the layer's weight dimensions
            outchannels = size(layer.weight, 1)  # The first dimension for Flux.Conv weight is the number of output channels
            
            kernel_size = (size(layer.weight, 3), size(layer.weight, 4))  # height and width of the kernel
            stride = layer.stride
            pad = layer.pad
            activation = layer.activation

            # Create a new convolutional layer with the adjusted number of input channels
            new_conv_layer = Flux.Conv(kernel_size, inchannels => outchannels, stride=stride, pad=pad, activation=activation)
            encoder_backbone.layers[index] = new_conv_layer
            break  
        end
    end
    return encoder_backbone
end


"""
    UNet(imsize::Dims{2} = (256, 256), inchannels::Integer = 3, outplanes::Integer = 3,
         encoder_backbone = Metalhead.backbone(DenseNet(121)); pretrain::Bool = false)

Creates a UNet model with an encoder built of specified backbone. By default it uses 
[`DenseNet`](@ref) backbone, but any ResNet-like Metalhead model can be used for the encoder.
([reference](https://arxiv.org/abs/1505.04597)).

# Arguments

  - `imsize`: size of input image
  - `inchannels`: number of channels in input image
  - `outplanes`: number of output feature planes.
  - `encoder_backbone`: The backbone layers of specified model to be used as encoder. For
    example, `Metalhead.backbone(Metalhead.ResNet(18))` can be passed to instantiate a UNet with layers of
    resnet18 as encoder.
  - `pretrain`: Whether to load the pre-trained weights for ImageNet

!!! warning

    `UNet` does not currently support pretrained weights.

See also [`Metalhead.unet`](@ref).
"""
struct UNet
    layers::Any
end
@functor UNet

function UNet(imsize::Dims{2} = (256, 256), inchannels::Integer = 3, outplanes::Integer = 3,
              encoder_backbone = Metalhead.backbone(DenseNet(121)); pretrain::Bool = false)
    # Modify the encoder backbone to adjust the first convolutional layer's input channels
    encoder_backbone = modify_first_conv_layer(encoder_backbone, inchannels)
    
    layers = unet(encoder_backbone, imsize, inchannels, outplanes)
    model = UNet(layers)
    
    if pretrain

        artifact_name = "UNet"
        loadpretrain!(model, artifact_name)
    end
    
    return model
end

(m::UNet)(x::AbstractArray) = m.layers(x)
