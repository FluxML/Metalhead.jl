include("inception_v3.jl")

struct Block35
    path_1
    path_2
    path_3
    path_4
end

@treelike Block35

function Block35(in_channels)
    path_1 = ConvBlock((1,1), in_channels=>32)

    path_2 = Chain(ConvBlock((1,1), in_channels=>32),
		ConvBlock((3,3), 32=>32, pad = (1,1)))

    path_3 = Chain(ConvBlock((1,1), in_channels=>32),
		ConvBlock((3,3), 32=>48, pad = (1,1)),
		ConvBlock((3,3), 48=>64, pad = (1,1)))

    path_4 = ConvBlock((1,1), 128=>320)

    Block35(path_1, path_2, path_3, path_4)
end

function (c::Block35)(x)
     op1 = c.path_1(x)
     op2 = c.path_2(x)
     op3 = c.path_3(x)
     op = cat(op1, op2, op3; dims = 3)
     c.path_4(op)
end

struct Block17
     path_1
     path_2
     path_3
end

@treelike Block17

function Block17(in_channels)
     path_1 = ConvBlock((1,1), in_channels=>192)

     path_2 = Chain(ConvBlock((1,1), in_channels=>128),
		ConvBlock((1,7), 128=>160, pad = (0,3)),
		ConvBlock((7,1), 160=>192, pad = (3,0)))

     path_3 = ConvBlock((1,1), 384=>1088)

     Block17(path_1, path_2, path_3)
end

function (c::Block17)(x)
     op1 = c.path_1(x)
     op2 = c.path_2(x)
     op = cat(op1, op2; dims = 3)
     c.path_3(op)
end

struct Block8
     path_1
     path_2
     path_3
end

@treelike Block8

function Block8(in_channels)
     path_1 = ConvBlock((1,1), in_channels=>192)

     path_2 = Chain(ConvBlock((1,1), in_channels=>192, pad = (1,1)),
		ConvBlock((1,1), 192=>192),
		ConvBlock((1,3), 192=>224),
		ConvBlock((3,1), 224=>256))

     path_3 = ConvBlock((1,1), 448=>2080)

     Block8(path_1, path_2, path_3)
end

function (c::Block8)(x)
     op1 = c.path_1(x)
     op2 = c.path_2(x)
     op = cat(op1, op2; dims=3)
     c.path_3(op)
end

struct Res35
     block35
end

@treelike Res35

function (c::Res35)(x, scale = 1.0f0)
     op = c.block35(x)
     b = op .+ (scale .* x)
     c = relu.(b)
     c
end

struct InceptionPureB
    path_1
    path_2
    path_3
end

@treelike InceptionPureB

function InceptionPureB(in_channels)
    path_1 = ConvBlock((3, 3), in_channels=>384, stride = (2, 2))

    path_2 = Chain(ConvBlock((1, 1), in_channels=>256),
              ConvBlock((3, 3), 256=>256, pad = (1, 1)),
              ConvBlock((3, 3), 256=>384, stride = (2, 2)))

    path_3 = x -> maxpool(x, (3, 3), stride = (2, 2))

    InceptionPureB(path_1, path_2, path_3)
end

function (c::InceptionPureB)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    cat(op1, op2, op3; dims = 3)
end

struct InceptionReductionB
    path_1
    path_2
    path_3
    path_4
end

@treelike InceptionReductionB

function InceptionReductionB(in_channels)
    path_1 = Chain(ConvBlock((1,1), in_channels=>256),
                 ConvBlock((3,3), 256=>384, stride = (2,2)))

    path_2 = Chain(ConvBlock((1,1), in_channels=>256),
                 ConvBlock((3,3), 256=>288, stride = (2,2)))

    path_3 = Chain(ConvBlock((1,1), in_channels=>256, pad = (1,1)),
		ConvBlock((3,3), 256=>288),
		ConvBlock((3,3), 288=>320, stride = (2,2)))

    path_4 = x -> maxpool(x, (3,3), stride = (2,2))

    InceptionReductionB(path_1, path_2, path_3, path_4)
end

function (c::InceptionReductionB)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    op4 = c.path_4(x)
    cat(op1, op2, op3, op4; dims=3)
end

struct Stem
    path_1
end

@treelike Stem

function Stem()
    path_1 = Chain(ConvBlock((3,3), 3=>32, stride = (2,2)),
		ConvBlock((3,3), 32=>32),
		ConvBlock((3,3), 32=>64),
		x -> maxpool(x, (3,3), stride = (2,2), pad = (1,1)),
		ConvBlock((1,1), 64=>80),
		ConvBlock((3,3), 80=>192),
		x -> maxpool(x, (3,3), stride = (2,2)))

    Stem(path_1)
end

function (c::Stem)(x)
    op1 = c.path_1(x)
    op1
end

struct InceptionPureA
    path_1
    path_2
    path_3
    path_4
end

@treelike InceptionPureA

function InceptionPureA(in_channels, pool_features)
    path_1 = ConvBlock((1, 1), in_channels=>96)

    path_2 = Chain(ConvBlock((1, 1), in_channels=>48),
              ConvBlock((5, 5), 48=>64, pad = (2,2)))

    path_3 = Chain(ConvBlock((1, 1), in_channels=>64),
              ConvBlock((3, 3), 64=>96, pad = (1,1)),
              ConvBlock((3, 3), 96=>96, pad = (1,1)))

    path_4 = Chain(x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
              ConvBlock((1, 1), in_channels=>pool_features))

    InceptionPureA(path_1, path_2, path_3, path_4)
end

function (c::InceptionPureA)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    op4 = c.path_4(x)
    cat(op1, op2, op3, op4; dims = 3)
end

inception_resnet() = Chain(Stem(),
			InceptionPureA(192, 64),
			[Res35(Block35(320)) for _ in 1:10]...,
			InceptionPureB(320),
			[Res35(Block17(1088)) for _ in 1:20]...,
			InceptionReductionB(1088),
			[Res35(Block8(2080)) for _ in 1:10]...,
			ConvBlock((1,1), 2080=>1536)) |> gpu

model = Chain(inception_resnet(),
		x -> meanpool(x, (8,8)),
		x -> x[1,1,:,:],
		Dense(1536, 1024),
		Dense(1024, 512),
		Dense(512, 128),
		Dense(128, 2),
		softmax) |> gpu;
