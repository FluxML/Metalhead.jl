mixed3_a() = Parallel(cat_channels,
                     MaxPool((3, 3); stride = 2),
                     Chain(conv_bn((3, 3), 64, 96; stride = 2)...))

mixed4_a() = Parallel(cat_channels,
                     Chain(conv_bn((1, 1), 160, 64)...,
                           conv_bn((3, 3), 64, 96)...),
                     Chain(conv_bn((1, 1), 160, 64)...,
                           conv_bn((1, 7), 64, 64; pad = (0, 3))...,
                           conv_bn((7, 1), 64, 64; pad = (3, 0))...,
                           conv_bn((3, 3), 64, 96)...))

mixed5_a() = Parallel(cat_channels,
                     Chain(conv_bn((3, 3), 192, 192; stride = 2)...),
                     MaxPool((3, 3); stride = 2))

function inception4_a()
	branch1 = Chain(conv_bn((1, 1), 384, 96)...)

	branch2 = Chain(conv_bn((1, 1), 384, 64)...,
									conv_bn((3, 3), 64, 96; pad = 1)...)

	branch3 = Chain(conv_bn((1, 1), 384, 64)...,
									conv_bn((3, 3), 64, 96; pad = 1)...,
									conv_bn((3, 3), 96, 96; pad = 1)...)

	branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_bn((1, 1), 384, 96)...)

	return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_a()
	branch1 = Chain(conv_bn((3, 3), 384, 384; stride = 2)...)

	branch2 = Chain(conv_bn((1, 1), 384, 192)...,
									conv_bn((3, 3), 192, 224; pad = 1)...,
									conv_bn((3, 3), 224, 256; stride = 2)...)

	branch3 = MaxPool((3, 3); stride = 2)

	return Parallel(cat_channels, branch1, branch2, branch3)
end

function inception4_b()
	branch1 = Chain(conv_bn((1, 1), 1024, 384)...)

	branch2 = Chain(conv_bn((1, 1), 1024, 192)...,
									conv_bn((1, 7), 192, 224; pad = (0, 3))...,
									conv_bn((7, 1), 224, 256; pad = (3, 0))...)

	branch3 = Chain(conv_bn((1, 1), 1024, 192)...,
									conv_bn((7, 1), 192, 192; pad = (0, 3))...,
									conv_bn((1, 7), 192, 224; pad = (3, 0))...,
									conv_bn((7, 1), 224, 224; pad = (0, 3))...,
									conv_bn((1, 7), 224, 256; pad = (3, 0))...)

	branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_bn((1, 1), 1024, 128)...)

	return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function reduction_b()
	branch1 = Chain(conv_bn((1, 1), 1024, 192)...,
									conv_bn((3, 3), 192, 192; stride = 2)...)

	branch2 = Chain(conv_bn((1, 1), 1024, 256)...,
									conv_bn((1, 7), 256, 256; pad = (0, 3))...,
									conv_bn((7, 1), 256, 320; pad = (3, 0))...,
									conv_bn((3, 3), 320, 320; stride = 2)...)

	branch3 = MaxPool((3, 3); stride = 2)

	return Parallel(cat_channels, branch1, branch2, branch3)
end

function inception4_c()
	branch1 = Chain(conv_bn((1, 1), 1536, 256)...)

	branch2 =  Chain(conv_bn((1, 1), 1536, 384)...,
									 Parallel(cat_channels,
														Chain(conv_bn((1, 3), 384, 256; pad = (0, 1))...),
														Chain(conv_bn((3, 1), 384, 256; pad = (1, 0))...)))

	branch3 = Chain(conv_bn((1, 1), 1536, 384)...,
									conv_bn((3, 1), 384, 448; pad = (1, 0))...,
									conv_bn((1, 3), 448, 512; pad = (0, 1))...,
									Parallel(cat_channels,
													 Chain(conv_bn((1, 3), 512, 256; pad = (0, 1))...),
													 Chain(conv_bn((3, 1), 512, 256; pad = (1, 0))...)))

	branch4 = Chain(MeanPool((3, 3); stride = 1, pad = 1), conv_bn((1, 1), 1536, 256)...)

	return Parallel(cat_channels, branch1, branch2, branch3, branch4)
end

function inception4(; nchannels = 3, dropout = 0., nclasses = 1000)
	body = Chain(conv_bn((3, 3), nchannels, 32; stride = 2)...,
							 conv_bn((3, 3), 32, 32; stride = 1)...,
							 conv_bn((3, 3), 32, 64; stride = 1, pad = 1)...,
							 mixed3_a(),
							 mixed4_a(),
							 mixed5_a(),
							 inception4_a(),
							 inception4_a(),
							 inception4_a(),
							 inception4_a(),
							 reduction_a(),  # mixed6a
							 inception4_b(),
							 inception4_b(),
							 inception4_b(),
							 inception4_b(),
							 inception4_b(),
							 inception4_b(),
							 inception4_b(),
							 reduction_b(),  # mixed7a
							 inception4_c(),
							 inception4_c(),
							 inception4_c())
	head = Chain(GlobalMeanPool(), MLUtils.flatten, Dropout(dropout), Dense(1536, nclasses))
	return Chain(body, head)
end

struct Inception4
	layers
end

function Inception4(; nchannels = 3, dropout = 0., nclasses = 1000)
	layers = inception4(; nchannels, dropout, nclasses)
	return Inception4(layers)
end

@functor Inception4

(m::Inception4)(x)  = m.layers(x)

backbone(m::Inception4) = m.layers[1]
classifier(m::Inception4) = m.layers[2]
