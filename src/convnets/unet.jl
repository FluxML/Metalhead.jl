function PixelShuffleICNR(inplanes, outplanes; r = 2)
    return Chain(basic_conv_bn((1, 1), inplanes, outplanes * (r^2)),
                 Flux.PixelShuffle(r))
end

function UNetCombineLayer(inplanes, outplanes)
    return Chain(basic_conv_bn((3, 3), inplanes, outplanes; pad = 1),
                 basic_conv_bn((3, 3), outplanes, outplanes; pad = 1))
end

function UNetMiddleBlock(inplanes)
    return Chain(basic_conv_bn((3, 3), inplanes, 2inplanes; pad = 1),
                 basic_conv_bn((3, 3), 2inplanes, inplanes; pad = 1))
end

function UNetFinalBlock(inplanes, outplanes)
    return Chain(basicblock(inplanes, inplanes; reduction_factor = 1),
                 basic_conv_bn((1, 1), inplanes, outplanes))
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

        return UNetBlock(Chain(layer, childunet),
                         inplanes, midplanes, outplanes)
    end
end

function UNetBlock(m_child, inplanes, midplanes, outplanes = 2inplanes)
    return Chain(;
                 upsample = SkipConnection(Chain(; child = m_child,
                                                 upsample = PixelShuffleICNR(midplanes,
                                                                             midplanes)),
                                           Parallel(cat_channels, identity,
                                                    BatchNorm(inplanes))),
                 act = xs -> relu.(xs),
                 combine = UNetCombineLayer(inplanes + midplanes, outplanes))
end

"""
    UNet(backbone, inputsize)
    DenseNet(transition_configs::NTuple{N,Integer})

Creates a UNet model with specified backbone. Backbone of Any Metalhead model can be used as
encoder .
Set `pretrain = true` to load the model with pre-trained weights for ImageNet.

!!! warning

    `UNet` does not currently support pretrained weights.

See also [`Metalhead.UNet`](@ref).
"""
struct UNet
    layers::Any
end
@functor UNet

function UNet(backbone,
              inputsize,
              outplanes,
              final = UNetFinalBlock,
              fdownscale::Integer = 0,
              kwargs...)
    backbonelayers = collect(iterlayers(backbone))
    layers = unetlayers(backbonelayers,
                        inputsize;
                        m_middle = UNetMiddleBlock,
                        skip_upscale = fdownscale,
                        kwargs...)

    outsz = Flux.outputsize(layers, inputsize)
    layers = Chain(layers, final(outsz[end - 1], outplanes))

    return UNet(layers)
end

(m::UNet)(x::AbstractArray) = m.layers(x)
