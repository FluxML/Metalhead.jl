function PixelShuffleICNR(inplanes, outplanes; r = 2)
	return Chain(basic_conv_bn((1, 1), inplanes, outplanes * (r^2)),
		Flux.PixelShuffle(r))
end

function UNetCombineLayer(inplanes, outplanes)
	return Chain(basic_conv_bn((3, 3), inplanes, outplanes; pad = 1),
		basic_conv_bn((3, 3), outplanes, outplanes; pad = 1))
end

function UNetMiddleBlock(inplanes)
	return Chain(basic_conv_bn((3, 3), inplanes, 2 * inplanes; pad = 1),
		basic_conv_bn((3, 3), 2 * inplanes, inplanes; pad = 1))
end

function UNetFinalBlock(inplanes, outplanes)
	return Chain(basicblock(inplanes, inplanes; reduction_factor = 1),
		basic_conv_bn((1, 1), inplanes, outplanes))
end

function unetlayers(layers, sz; outplanes = nothing, skip_upscale = 0,
	m_middle = _ -> (identity,))
	isempty(layers) && return m_middle(sz[end-1])

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

		inplanes = sz[end-1]
		midplanes = outsz[end-1]
		outplanes = isnothing(outplanes) ? inplanes : outplanes

		return UNetBlock(Chain(layer, childunet),
			inplanes, midplanes, outplanes)
	end
end

function UNetBlock(m_child, inplanes, midplanes, outplanes = 2 * inplanes)
	return Chain(SkipConnection(Chain(m_child,
				PixelShuffleICNR(midplanes, midplanes)),
			Parallel(cat_channels, identity, BatchNorm(inplanes))),
		xs -> relu.(xs),
		UNetCombineLayer(inplanes + midplanes, outplanes))
end

"""
	unet(backbone::Vector{Any}; inputsize, outplanes::Integer = 3,
		final::Any = UNetFinalBlock, fdownscale::Integer = 0, kwargs...)

Creates a UNet model with specified backbone. Backbone of Any Metalhead model 
can be used as encoder. 
([reference](https://arxiv.org/abs/1505.04597)).

# Arguments

	- `backbone`: The backbone layers to be used in encoder. 
		For example, `Metalhead.backbone(Metalhead.ResNet(18))` can be passed 
		to instantiate a UNet with layers of resnet18 as encoder.
	- `inputsize`: size of input image
	- `outplanes`: number of output feature planes
	- `final`: final block as described in original paper
	- `fdownscale`: downscale factor
"""
function unet(backbone::Vector{Any}, inputsize, outplanes::Integer,
	final::Any = UNetFinalBlock, fdownscale::Integer = 0, kwargs...)

	backbonelayers = collect(iterlayers(backbone))
	layers = unetlayers(backbonelayers, inputsize; m_middle = UNetMiddleBlock,
		skip_upscale = fdownscale, kwargs...)

	outsz = Flux.outputsize(layers, inputsize)
	layers = Chain(layers, final(outsz[end-1], outplanes))
	
	return layers
end

"""
	UNet(backbone::Vector{Any}; pretrain::Bool = false, inputsize::NTuple{4, Integer}, 
	outchannels::Integer = 3)

Creates a UNet model with specified backbone. Backbone of Any Metalhead model can be used as
encoder. 
([reference](https://arxiv.org/abs/1505.04597)).

# Arguments

	- `backbone`: The backbone layers to be used in encoder. 
		For example, `Metalhead.backbone(Metalhead.ResNet(18))` can be passed to instantiate a UNet with layers of
		resnet18 as encoder.
	- `pretrain`: Whether to load the pre-trained weights for ImageNet
	- `inputsize`: size of input image
	- `outchannels`: number of output channels.

!!! warning

	`UNet` does not currently support pretrained weights.

See also [`Metalhead.UNet`](@ref).
"""
struct UNet
	layers::Any
end
@functor UNet

function UNet(backbone::Vector{Any}; pretrain::Bool = false, inputsize::NTuple{4, Integer},
	outchannels::Integer = 3)
	layers = unet(backbone, inputsize, outchannels)
    if pretrain
        loadpretrain!(layers, string("UNet"))
    end
	return UNet(layers)
end

(m::UNet)(x::AbstractArray) = m.layers(x)
