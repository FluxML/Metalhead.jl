function squeezenet_layers()
  weight = Metalhead.weights("squeezenet.bson")
  weights = Dict{Any ,Any}()
  for ele in keys(weight)
    weights[string(ele)] = weight[ele]
  end
    c_1 = Conv(weights["conv10_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["conv10_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_2 = Dropout(0.5f0)
    c_3 = Conv(weights["fire9/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire9/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_4 = Conv(weights["fire9/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire9/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_5 = Conv(weights["fire8/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire8/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_6 = Conv(weights["fire8/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire8/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_7 = Conv(weights["fire7/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire7/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_8 = Conv(weights["fire7/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire7/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_9 = Conv(weights["fire6/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire6/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_10 = Conv(weights["fire6/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire6/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_11 = Conv(weights["fire5/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire5/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_12 = Conv(weights["fire5/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire5/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_13 = Conv(weights["fire4/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire4/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_14 = Conv(weights["fire4/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire4/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_15 = Conv(weights["fire3/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire3/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_16 = Conv(weights["fire3/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire3/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_17 = Conv(weights["fire2/expand1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire2/expand1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_18 = Conv(weights["fire2/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire2/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1))
    c_19 = Conv(weights["conv1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["conv1_b_0"], stride=(2, 2), pad=(0, 0), dilation = (1, 1))
    c_20 = Conv(weights["fire2/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire2/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_21 = Conv(weights["fire3/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire3/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_22 = Conv(weights["fire4/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire4/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_23 = Conv(weights["fire5/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire5/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_24 = Conv(weights["fire6/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire6/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_25 = Conv(weights["fire7/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire7/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_26 = Conv(weights["fire8/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire8/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))
    c_27 = Conv(weights["fire9/expand3x3_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire9/expand3x3_b_0"], stride=(1, 1), pad=(1, 1), dilation = (1, 1))

    ls = Chain(Conv(weights["conv1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["conv1_b_0"], stride=(2, 2), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), MaxPool((3,3), pad=(0,0), stride=(2,2)),
            Conv(weights["fire2/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire2/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_17(x)), relu.(c_20(x)), dims=3),
            Conv(weights["fire3/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire3/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_15(x)), relu.(c_21(x)), dims=3),
            MaxPool((3, 3), pad=(0, 0), stride=(2, 2)),
            Conv(weights["fire4/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire4/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_13(x)), relu.(c_22(x)), dims=3),
            Conv(weights["fire5/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire5/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_11(x)), relu.(c_23(x)), dims=3),
            MaxPool((3, 3), pad=(0, 0), stride=(2, 2)),
            Conv(weights["fire6/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire6/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_9(x)), relu.(c_24(x)), dims=3),
            Conv(weights["fire7/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire7/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_7(x)), relu.(c_25(x)), dims=3),
            Conv(weights["fire8/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire8/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_5(x)), relu.(c_26(x)), dims=3),
            Conv(weights["fire9/squeeze1x1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["fire9/squeeze1x1_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->cat(relu.(c_3(x)), relu.(c_27(x)), dims=3),
            Dropout(0.5f0),
            Conv(weights["conv10_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:], weights["conv10_b_0"], stride=(1, 1), pad=(0, 0), dilation = (1, 1)),
            x -> relu.(x), x->mean(x, dims=[1,2]),
            vec, softmax
            )
#end
  return ls
end

struct SqueezeNet <: ClassificationModel{ImageNet.ImageNet1k}
  layers::Chain
end

SqueezeNet() = SqueezeNet(squeezenet_layers())

Base.show(io::IO, ::SqueezeNet) = print(io, "SqueezeNet()")

@functor SqueezeNet

(m::SqueezeNet)(x) = m.layers(x)
