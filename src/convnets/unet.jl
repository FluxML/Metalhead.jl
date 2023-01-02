function unet_block(in_chs::Int, out_chs::Int, kernel = (3, 3))
	return Chain(; conv1 = Conv(kernel, in_chs => out_chs; pad = (1, 1)),
		norm1 = BatchNorm(out_chs, relu),
		conv2 = Conv(kernel, out_chs => out_chs; pad = (1, 1)),
		norm2 = BatchNorm(out_chs, relu))
end

function upconv_block(in_chs::Int, out_chs::Int, kernel = (2, 2))
	return ConvTranspose(kernel, in_chs => out_chs; stride = (2, 2))
end

function unet(in_channels::Integer = 3, out_channels::Integer = in_channels,
	features::Integer = 32)
	encoder_conv = []
	push!(encoder_conv,
		unet_block(in_channels, features))

	append!(encoder_conv,
		[unet_block(features * 2^i, features * 2^(i + 1)) for i in 0:2])

	encoder_conv = Chain(encoder_conv)
	encoder_pool = [Chain(encoder_conv[i], MaxPool((2, 2); stride = (2, 2))) for i in 1:4]

	bottleneck = unet_block(features * 8, features * 16)
	layers = Chain(encoder_conv, bottleneck)

	upconv = Chain([upconv_block(features * 2^(i + 1), features * 2^i)
					for i in 0:3]...)

	concat_layer = Chain([Parallel(cat_channels,
		encoder_pool[i],
		upconv[i])
						  for i in 1:4]...)

	decoder_layer = Chain([unet_block(features * 2^(i + 1), features * 2^i) for i in 3:-1:0]...)

	layers = Chain(layers, decoder_layer)

	decoder = Chain([Chain([
		concat_layer[i],
		decoder_layer[5-i]])
					 for i in 4:-1:1]...)

	final_conv = Conv((1, 1), features => out_channels, σ)

	decoder = Chain(decoder, final_conv)

	return layers
end

"""
	UNet(in_channels::Integer = 3, inplanes::Integer = 32, outplanes::Integer = inplanes)

	Create a UNet model
	([reference](https://arxiv.org/abs/1505.04597v1))

	# Arguments
	- `in_channels`: The number of input channels
	- `inplanes`: The number of input features to the network
	- `outplanes`: The number of output features

!!! warning

	`UNet` does not currently support pretrained weights.
"""
struct UNet
	layers::Any
end
@functor UNet

function UNet(in_channels::Integer = 3, inplanes::Integer = 32,
	outplanes::Integer = inplanes)
	layers = unet(in_channels, inplanes, outplanes)
	return UNet(layers)
end

(m::UNet)(x::AbstractArray) = m.layers(x)
