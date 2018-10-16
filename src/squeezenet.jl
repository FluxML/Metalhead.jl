struct Fire
  squeeze
  expand1x1
  expand3x3
end

@treelike Fire

Fire(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes) =
  Fire(Conv((1, 1), inplanes=>squeeze_planes, relu),
       Conv((1, 1), squeeze_planes=>expand1x1_planes, relu),
       Conv((3, 3), squeeze_planes=>expand3x3_planes, relu, pad=(1, 1)))

function (f::Fire)(x)
  x = f.squeeze(x)
  cat(f.expand1x1(x), f.expand3x3(x), dims=3)
end

# NOTE: The initialization of the Conv layers are different in the paper. They are Kaiming Normal
load_squeezenetv1_0() = Chain(Conv((7, 7), 3=>96, relu, stride = (2, 2)),
    MaxPool((3, 3), stride = (2, 2)),
    Fire(96, 16, 64, 64),
    Fire(128, 16, 64, 64),
    Fire(128, 32, 128, 128),
    MaxPool((3, 3), stride = (2, 2)),
    Fire(256, 32, 128, 128),
    Fire(256, 48, 192, 192),
    Fire(384, 48, 192, 192),
    Fire(384, 64, 256, 256),
    MaxPool((3, 3), stride = (2, 2)),
    Fire(512, 64, 256, 256),
    Dropout(0.5),
    Conv((1, 1), 512=>1000, relu),
    MeanPool((12, 12), stride = (1, 1)),
    x -> reshape(x, :, size(x, 4)),
    softmax)

load_squeezenetv1_1() = Chain(Conv((3, 3), 3=>64, relu, stride = (2, 2)),
    MaxPool((3, 3), stride = (2, 2)),
    Fire(64, 16, 64, 64),
    Fire(128, 16, 64, 64),
    MaxPool((3, 3), stride = (2, 2)),
    Fire(128, 32, 128, 128),
    Fire(256, 32, 128, 128),
    MaxPool((3, 3), stride = (2, 2)),
    Fire(256, 48, 192, 192),
    Fire(384, 48, 192, 192),
    Fire(384, 64, 256, 256),
    Fire(512, 64, 256, 256),
    Dropout(0.5),
    Conv((1, 1), 512=>1000, relu),
    MeanPool((13, 13), stride = (1, 1)),
    x -> reshape(x, :, size(x, 4)),
    softmax)

function trained_squeezenetv1_1_layers()
  weight = Metalhead.weights("squeezenet.bson")
  weights = Dict{Any ,Any}()
  for ele in keys(weight)
    weights[string(ele)] = weight[ele]
  end
    c_1 = Conv(flipkernel(weights["conv10_w_0"]), weights["conv10_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_2 = Dropout(0.5f0)
    c_3 = Conv(flipkernel(weights["fire9/expand1x1_w_0"]), weights["fire9/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_4 = Conv(flipkernel(weights["fire9/squeeze1x1_w_0"]), weights["fire9/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_5 = Conv(flipkernel(weights["fire8/expand1x1_w_0"]), weights["fire8/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_6 = Conv(flipkernel(weights["fire8/squeeze1x1_w_0"]), weights["fire8/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_7 = Conv(flipkernel(weights["fire7/expand1x1_w_0"]), weights["fire7/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_8 = Conv(flipkernel(weights["fire7/squeeze1x1_w_0"]), weights["fire7/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_9 = Conv(flipkernel(weights["fire6/expand1x1_w_0"]), weights["fire6/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_10 = Conv(flipkernel(weights["fire6/squeeze1x1_w_0"]), weights["fire6/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_11 = Conv(flipkernel(weights["fire5/expand1x1_w_0"]), weights["fire5/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_12 = Conv(flipkernel(weights["fire5/squeeze1x1_w_0"]), weights["fire5/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_13 = Conv(flipkernel(weights["fire4/expand1x1_w_0"]), weights["fire4/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_14 = Conv(flipkernel(weights["fire4/squeeze1x1_w_0"]), weights["fire4/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_15 = Conv(flipkernel(weights["fire3/expand1x1_w_0"]), weights["fire3/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_16 = Conv(flipkernel(weights["fire3/squeeze1x1_w_0"]), weights["fire3/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_17 = Conv(flipkernel(weights["fire2/expand1x1_w_0"]), weights["fire2/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_18 = Conv(flipkernel(weights["fire2/squeeze1x1_w_0"]), weights["fire2/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_19 = Conv(flipkernel(weights["conv1_w_0"]), weights["conv1_b_0"], stride=(2, 2), pad=(0, 0), dilation = (1, 1))
    c_20 = Conv(flipkernel(weights["fire2/expand3x3_w_0"]), weights["fire2/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_21 = Conv(flipkernel(weights["fire3/expand3x3_w_0"]), weights["fire3/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_22 = Conv(flipkernel(weights["fire4/expand3x3_w_0"]), weights["fire4/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_23 = Conv(flipkernel(weights["fire5/expand3x3_w_0"]), weights["fire5/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_24 = Conv(flipkernel(weights["fire6/expand3x3_w_0"]), weights["fire6/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_25 = Conv(flipkernel(weights["fire7/expand3x3_w_0"]), weights["fire7/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_26 = Conv(flipkernel(weights["fire8/expand3x3_w_0"]), weights["fire8/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_27 = Conv(flipkernel(weights["fire9/expand3x3_w_0"]), weights["fire9/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))

    ls = Chain(Conv(flipkernel(weights["conv1_w_0"]), weights["conv1_b_0"], stride=(2, 2), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), MaxPool((3,3), pad=(0,0), stride=(2,2)),
            Conv(flipkernel(weights["fire2/squeeze1x1_w_0"]), weights["fire2/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_17(x)), relu.(c_20(x)), dims=3),
            Conv(flipkernel(weights["fire3/squeeze1x1_w_0"]), weights["fire3/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_15(x)), relu.(c_21(x)), dims=3),
            MaxPool((3, 3), pad=(0, 0), stride=(2, 2)),
            Conv(flipkernel(weights["fire4/squeeze1x1_w_0"]), weights["fire4/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_13(x)), relu.(c_22(x)), dims=3),
            Conv(flipkernel(weights["fire5/squeeze1x1_w_0"]), weights["fire5/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_11(x)), relu.(c_23(x)), dims=3),
            MaxPool((3, 3), pad=(0, 0), stride=(2, 2)),
            Conv(flipkernel(weights["fire6/squeeze1x1_w_0"]), weights["fire6/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_9(x)), relu.(c_24(x)), dims=3),
            Conv(flipkernel(weights["fire7/squeeze1x1_w_0"]), weights["fire7/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_7(x)), relu.(c_25(x)), dims=3),
            Conv(flipkernel(weights["fire8/squeeze1x1_w_0"]), weights["fire8/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_5(x)), relu.(c_26(x)), dims=3),
            Conv(flipkernel(weights["fire9/squeeze1x1_w_0"]), weights["fire9/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_3(x)), relu.(c_27(x)), dims=3),
            Dropout(0.5f0),
            Conv(flipkernel(weights["conv10_w_0"]), weights["conv10_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->mean(x, dims=[1,2]),
            x -> reshape(x, :, size(x, 4)), softmax
            )
  Flux.testmode!(ls)
  return ls
end

struct SqueezeNet <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

function SqueezeNet(version::String = "1.1")
  if version == "1.0"
    SqueezeNet(load_squeezenetv1_0())
  elseif version == "1.1"
    SqueezeNet(load_squeezenetv1_1())
  else
    error("Only SqueezeNet versions 1.1 and 1.0 available")
  end
end

function trained(::Type{SqueezeNet}, version = "1.1")
  if version == "1.0"
    error("Pretrained Weights for SqueezeNet v1.0 are not available")
  elseif version == "1.1"
    SqueezeNet(trained_squeezenetv1_1_layers())
  else
    error("Only SqueezeNet versions 1.1 and 1.0 available")
  end
end

Base.show(io::IO, ::SqueezeNet) = print(io, "SqueezeNet()")

@treelike SqueezeNet

(m::SqueezeNet)(x) = m.layers(x)
