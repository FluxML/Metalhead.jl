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

struct Unet
	encoder::Any
	decoder::Any
	upconv::Any
	pool::Any
	bottleneck::Any
	final_conv::Any
end

@functor Unet

function Unet(inplanes::Int = 3, outplanes::Int = 1, init_features::Int = 32)

	features = init_features

	encoder_layers = []
	append!(encoder_layers, [unet_block(inplanes, features)])
	append!(encoder_layers, [unet_block(features * 2^i, features * 2^(i + 1)) for i ∈ 0:2])

	encoder = Chain(encoder_layers)

	decoder = Chain([unet_block(features * 2^(i + 1), features * 2^i) for i ∈ 0:3])

	pool = Chain([MaxPool((2, 2), stride = (2, 2)) for _ ∈ 1:4])

	upconv = Chain([UpConvBlock(features * 2^(i + 1), features * 2^i) for i ∈ 3:-1:0])

	bottleneck = _block(features * 8, features * 16)

	final_conv = Conv((1, 1), features => outplanes)

	Unet(encoder, decoder, upconv, pool, bottleneck, final_conv)
end

function (u::Unet)(x::AbstractArray)
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
