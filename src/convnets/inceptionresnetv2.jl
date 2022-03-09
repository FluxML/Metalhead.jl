conv2d(inc, out, k, p, s = 1, b = true) = Chain(conv_bn(k, inc, out, identity; ϵ = 1e-3, momentum = 0.1, stride = s, pad = p, bias = b)...)

function reduction_a(inc, k, l, m, n)
    branch_0 = Chain(conv2d(inc, n, (3, 3), 0, 2, false))
    branch_1 = Chain(conv2d(inc, k, (1, 1), 0, 1, false),
                     conv2d(k, l, (3, 3), 1, 1, false),
                     conv2d(l, m, (3, 3), 0, 2, false))
    branch_2 = Chain(MaxPool((3, 3), stride = 2))

    Parallel(cat_channels,
             branch_0, branch_1, branch_2)
end

function stem(inc)
    features = Chain(conv2d(inc, 32, (3, 3), 0, 2, false),
                     conv2d(32, 32, (3, 3), 0, 1, false),
                     conv2d(32, 64, (3, 3), 1, 1, false),
                     MaxPool((3, 3), stride = 2),
                     conv2d(64, 80, (1, 1), 0, 1, false),
                     conv2d(80, 192, (1, 1), 0, 1, false),
                     MaxPool((3, 3), stride = 2))
    branch_0 = Chain(conv2d(192, 96, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(192, 48, (1, 1), 0, 1, false),
                     conv2d(48, 64, (5, 5), 2, 1, false))
    branch_2 = Chain(conv2d(192, 64, (1, 1), 0, 1, false),
                     conv2d(64, 96, (3, 3), 1, 1, false),
                     conv2d(96, 96, (3, 3), 1, 1, false))
    branch_3 = Chain(MeanPool((3, 3), pad = 1, stride = (1, 1)),
                     conv2d(192, 64, (1, 1), 0, 1, false))
    
    Chain(features, Parallel(cat_channels,
                             branch_0, branch_1, branch_2, branch_3))
end

function inception_resnet_a(inc, scale = 1.0)
    branch_0 = Chain(conv2d(inc, 32, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(inc, 32, (1, 1), 0, 1, false),
                     conv2d(32, 32, (3, 3), 1, 1, false))
    branch_2 = Chain(conv2d(inc, 32, (1, 1), 0, 1, false),
                     conv2d(32, 48, (3, 3), 1, 1, false),
                     conv2d(48, 64, (3, 3), 1, 1, false))
    conv = Chain(Conv((1, 1), 128 => 320))
    

    Chain(SkipConnection(Chain(Parallel(cat_channels, branch_0, branch_1, branch_2), conv, x -> x * scale), +), x -> relu.(x))
end

function inception_resnet_b(inc, scale = 1.0)
    branch_0 = Chain(conv2d(inc, 192, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(inc, 128, (1, 1), 0, 1, false),
                     conv2d(128, 160, (1, 7), (0, 3), 1, false),
                     conv2d(160, 192, (7, 1), (3, 0), 1, false))
    conv = Conv((1, 1), 384 => 1088)
    
    
    Chain(SkipConnection(Chain(Parallel(cat_channels, branch_0, branch_1), conv, x -> x * scale), +), x -> relu.(x))
end

function reduction_b(inc)
    branch_0 = Chain(conv2d(inc, 256, (1, 1), 0, 1, false),
                     conv2d(256, 384, (3, 3), 0, 2, false))
    branch_1 = Chain(conv2d(inc, 256, (1, 1), 0, 1, false),
                     conv2d(256, 288, (3, 3), 0, 2, false))
    branch_2 = Chain(conv2d(inc, 256, (1, 1), 0, 1, false),
                     conv2d(256, 288, (3, 3), 1, 1, false),
                     conv2d(288, 320, (3, 3), 0, 2, false))
    branch_3 = Chain(MaxPool((3, 3), stride = 2))

    Parallel(cat_channels, 
             branch_0, branch_1, branch_2, branch_3)
end

function inception_resnet_c(inc, scale = 1.0; activation = true)
    branch_0 = Chain(conv2d(inc, 192, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(inc, 192, (1, 1), 0, 1, false),
                     conv2d(192, 224, (1, 3), (0, 1), 1, false),
                     conv2d(224, 256, (3, 1), (1, 0), 1, false))
    conv = Chain(conv2d(448, 2080, (1, 1), 0, 1, false))
    
    
    activation ? Chain(SkipConnection(Chain(Parallel(cat_channels, branch_0, branch_1), conv, x -> x * scale), +), x -> relu.(x)) : SkipConnection(Chain(Parallel(cat_channels, branch_0, branch_1), conv, x -> x * scale), +)
end

function inception_resnet_v2(inc = 3, classes = 1000, k = 256, l = 256, m = 384, n = 384)
    Chain(stem(inc), 
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          inception_resnet_a(320, 0.17),
          reduction_a(320, k, l, m, n),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          inception_resnet_b(1088, 0.10),
          reduction_b(1088),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, 0.20),
          inception_resnet_c(2080, activation = false),
          conv2d(2080, 1536, (1, 1), 0, 1, false),
          AdaptiveMeanPool((1, 1)),
          Flux.flatten,
          Dense(1536, classes))
end