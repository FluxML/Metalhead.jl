using Flux
using CuArrays
using Flux: onehotbatch, argmax, @epochs
using Base.Iterators: partition
using BSON: @save, @load
import Flux.@treelike

struct ConvBlock
    convlayer
    norm
    nonlinearity
end

@treelike ConvBlock

ConvBlock(kernel, chs; stride = (1, 1), pad = (0, 0)) = ConvBlock(Conv(kernel, chs, stride = stride, pad = pad),
                                                                  BatchNorm(chs[2]),
                                                                  x -> relu.(x))

(c::ConvBlock)(x) = c.nonlinearity(c.norm(c.convlayer(x)))

struct InceptionA
    path_1
    path_2
    path_3
    path_4
end

@treelike InceptionA

function InceptionA(in_channels, pool_features)
    path_1 = ConvBlock((1, 1), in_channels=>96)

    path_2 = Chain(ConvBlock((1, 1), in_channels=>64),
              ConvBlock((3, 3), 64=>96, pad = (1,1)))

    path_3 = Chain(ConvBlock((1, 1), in_channels=>64),
              ConvBlock((3, 3), 64=>96, pad = (1,1)),
              ConvBlock((3, 3), 96=>96, pad = (1,1)))

    path_4 = Chain(x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
              ConvBlock((1, 1), in_channels=>pool_features))

    InceptionA(path_1, path_2, path_3, path_4)
end

function (c::InceptionA)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    op4 = c.path_4(x)
    cat(op1, op2, op3, op4; dims = 3)
end

struct InceptionB
    path_1
    path_2
    path_3
end

@treelike InceptionB

function InceptionB(in_channels)
    path_1 = ConvBlock((3, 3), in_channels=>384, stride = (2, 2))

    path_2 = Chain(ConvBlock((1, 1), in_channels=>192),
              ConvBlock((3, 3), 192=>224, pad = (1, 1)),
              ConvBlock((3, 3), 224=>256, stride = (2, 2)))

    path_3 = x -> maxpool(x, (3, 3), stride = (2, 2))

    InceptionB(path_1, path_2, path_3)
end

function (c::InceptionB)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    cat(op1, op2, op3; dims = 3)
end

struct InceptionC
    path_1
    path_2
    path_3
    path_4
end

@treelike InceptionC

function InceptionC(in_channels)
    path_1 = ConvBlock((1, 1), in_channels=>384)

    path_2 = Chain(ConvBlock((1, 1), in_channels=>192),
              ConvBlock((1, 7), 192=>224, pad = (0, 3)),
              ConvBlock((7, 1), 224=>256, pad = (3, 0)))

    path_3 = Chain(ConvBlock((1, 1), in_channels=>192),
              ConvBlock((7, 1), 192=>192, pad = (3, 0)),
              ConvBlock((1, 7), 192=>224, pad = (0, 3)),
              ConvBlock((7, 1), 224=>224, pad = (3, 0)),
              ConvBlock((1, 7), 224=>256, pad = (0, 3)))

    path_4 = Chain(x -> meanpool(x, (3, 3), stride = (1, 1), pad = (1, 1)),
              ConvBlock((1, 1), in_channels=>128))

    InceptionC(path_1, path_2, path_3, path_4)
end

function (c::InceptionC)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    op4 = c.path_4(x)
    cat(op1, op2, op3, op4; dims = 3)
end

struct InceptionD
    path_1
    path_2
    path_3
end

@treelike InceptionD

function InceptionD(in_channels)
    path_1 = Chain(ConvBlock((1, 1), in_channels=>192),
              ConvBlock((3, 3), 192=>192, stride = (2, 2)))

    path_2 = Chain(ConvBlock((1, 1), in_channels=>256),
              ConvBlock((1, 7), 256=>256),
              ConvBlock((7, 1), 256=>320),
              ConvBlock((3, 3), 320=>320, stride = (2, 2), pad = (3,3)))

    path_3 = x -> maxpool(x, (3, 3), stride = (2, 2))

    InceptionD(path_1, path_2, path_3)
end

function (c::InceptionD)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    op3 = c.path_3(x)
    cat(op1, op2, op3; dims = 3)
end

struct InceptionE
    path_1
    path_2
    path_3
    path_4
end

@treelike InceptionE

struct InceptionE_0
     path_1
end

@treelike InceptionE_0

function InceptionE_0(in_channels)
     ConvBlock((1,1), in_channels=>256)
end

function (c::InceptionE_0)(x)
     c.path_1(x)
end

struct InceptionE_1
     base
     path_1
     path_2
end

@treelike InceptionE_1

function (c::InceptionE_1)(x)
    op1 = c.path_1(c.base(x))
    op2 = c.path_2(c.base(x))
    cat(op1, op2; dims = 3)
end

struct InceptionE_2
     base
     path_1
     path_2
end

@treelike InceptionE_2

function (c::InceptionE_2)(x)
     op1 = c.path_1(c.base(x))
     op2 = c.path_2(c.base(x))
     cat(op1, op2; dims = 3)
end

function InceptionE(in_channels)
     path_1 = InceptionE_0(in_channels)

     path_2_base = ConvBlock((1,1), in_channels=>384)
     path_2_1 = ConvBlock((1,3), 384=>256, pad = (0,1))
     path_2_2 = ConvBlock((3,1), 384=>256, pad = (1,0))
     path_2 = InceptionE_1(path_2_base,
			path_2_1,
			path_2_2)

     path_3_base = Chain(ConvBlock((1,1), in_channels=>384),
			ConvBlock((3,1), 384=>448),
			ConvBlock((1,3), 448=>512))
     path_3_1 = ConvBlock((1,3), 512=>256, pad = (1,2))
     path_3_2 = ConvBlock((3,1), 512=>256, pad = (2,1))
     path_3 = InceptionE_2(path_3_base, path_3_1, path_3_2)

     path_4 = Chain(x -> meanpool(x, (3,3), stride = (1,1), pad = (1,1)),
             ConvBlock((1,1), 1536=>256))

     InceptionE(path_1, path_2, path_3, path_4)
end


function (c::InceptionE)(x)
     op1 = c.path_1(x)
     op2 = c.path_2(x)
     op3 = c.path_3(x)
     op4 = c.path_4(x)
     cat(op1, op2, op3, op4; dims = 3)
end

########## Start Compiling Net

function Base_v4_0()
    Chain(ConvBlock((3, 3), 3=>32, stride = (2, 2), pad = (1,1)),
         ConvBlock((3, 3), 32=>32, pad = (1,1)),
         ConvBlock((3, 3), 32=>64, pad = (1, 1)))
end

struct Base_v4_1
     path_1
     path_2
end

function Base_v4_1(in_classes)
     path_1 = x -> maxpool(x, (3,3), stride = (2, 2))
     path_2 = ConvBlock((3,3), in_classes=>64, stride = (2,2))
     Base_v4_1(path_1, path_2)
end

function (c::Base_v4_1)(x)
     op1 = c.path_1(x)
     op2 = c.path_2(x)
     cat(op1, op2; dims = 3)
end

@treelike Base_v4_1

struct Base_v4_2
    path_1
    path_2
end

function Base_v4_2(in_classes)
     path_1 = Chain(ConvBlock((1,1), in_classes=>64),
             ConvBlock((3,3), 64=>96))

     path_2 = Chain(ConvBlock((1,1), in_classes=>64),
             ConvBlock((1,7), 64=>64),
             ConvBlock((7,1), 64=>64),
             ConvBlock((3,3), 64=>96, pad = (3,3)))
     Base_v4_2(path_1, path_2)
end

@treelike Base_v4_2

function (c::Base_v4_2)(x)
     op1 = c.path_1(x)
     op2 = c.path_2(x)
     cat(op1, op2; dims = 3)
end

struct Base_v4_3
    path_1
    path_2
end

function Base_v4_3(in_classes)
    path_1 = ConvBlock((3,3), in_classes=>192, stride = (2,2))

    path_2 = x -> maxpool(x, (3,3), stride = (2,2))
    Base_v4_3(path_1, path_2)
end

@treelike Base_v4_3

function (c::Base_v4_3)(x)
    op1 = c.path_1(x)
    op2 = c.path_2(x)
    cat(op1, op2; dims = 3)
end

inception_base() = Chain(
			Base_v4_0(),
			Base_v4_1(64),
			Base_v4_2(128),
			Base_v4_3(192))			

inception_v4() = Chain(
		inception_base(),
		[InceptionA(384, 96) for _ in 1:3]...,
		InceptionB(384),
		[InceptionC(1024) for _ in 1:7]...,
		InceptionD(1024),
		[InceptionE(1536) for _ in 1:3]...,
		x -> meanpool(x, (8,8), stride = (1,1)))


model = Chain(inception_v4(),
		x -> x[1,1,:,:],
		Dense(1536, 512),
		Dense(512, 128),
		Dense(128, 32),
		Dense(32, 2),
		softmax) |> gpu

opt = ADAM(params(model))
