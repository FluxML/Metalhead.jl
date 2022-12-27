function unet_block(in_chs::Int, out_chs::Int, kernel = (3, 3))
	Chain(conv1 = Conv(kernel, in_chs => out_chs, pad = (1, 1); init = _random_normal),
		norm1 = BatchNorm(out_chs, relu),
		conv2 = Conv(kernel, out_chs => out_chs, pad = (1, 1); init = _random_normal),
		norm2 = BatchNorm(out_chs, relu))
end

function UpConvBlock(in_chs::Int, out_chs::Int, kernel = (2, 2))
	Chain(convtranspose = ConvTranspose(kernel, in_chs => out_chs, stride = (2, 2); init = _random_normal),
		norm = BatchNorm(out_chs, relu))
end

"""
	UNet(inplanes::Integer = 3, outplanes::Integer = 1, init_features::Integer = 32)

	Create a UNet model
	([reference](https://arxiv.org/abs/1505.04597v1))

	# Arguments
	- `in_channels`: The number of input channels
	- `inplanes`: The number of input planes to the network
	- `outplanes`: The number of output features

!!! warning
	
	`UNet` does not currently support pretrained weights.
"""
struct UNet
	encoder::Any
	decoder::Any
	upconv::Any
	pool::Any
	bottleneck::Any
	final_conv::Any
end
@functor UNet

function UNet(in_channels::Integer = 3, inplanes::Integer = 32, outplanes::Integer = 1)

	features = inplanes

	encoder_layers = []
	append!(encoder_layers, [unet_block(in_channels, features)])
	append!(encoder_layers, [unet_block(features * 2^i, features * 2^(i + 1)) for i ∈ 0:2])

	encoder = Chain(encoder_layers)

	decoder = Chain([unet_block(features * 2^(i + 1), features * 2^i) for i ∈ 0:3])

	pool = Chain([MaxPool((2, 2), stride = (2, 2)) for _ ∈ 1:4])

	upconv = Chain([UpConvBlock(features * 2^(i + 1), features * 2^i) for i ∈ 3:-1:0])

	bottleneck = _block(features * 8, features * 16)

	final_conv = Conv((1, 1), features => outplanes)

	UNet(encoder, decoder, upconv, pool, bottleneck, final_conv)
end

function (u::UNet)(x::AbstractArray)
	enc_out = []

	out = x
	for i ∈ 1:4
		out = u.encoder[i](out)
		push!(enc_out, out)

		out = u.pool[i](out)
	end

	out = u.bottleneck(out)

	for i ∈ 4:-1:1
		out = u.upconv[5-i](out)
		out = cat(out, enc_out[i], dims = 3)
		out = u.decoder[i](out)
	end

	σ(u.final_conv(out))
end
