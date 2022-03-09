conv2d(inc, out, k, p, s = 1, b = true) = Chain(conv_bn(k, inc, out, identity; ϵ = 1e-3, momentum = 0.1, stride = s, pad = p, bias = b)...)

function stem(inc)
    conv2d_1a_3x3 = Chain(conv2d(inc, 32, (3, 3), 0, 2, false))
    conv2d_2a_3x3 = Chain(conv2d(32, 32, (3, 3), 0, 1, false))
    conv2d_2b_3x3 = Chain(conv2d(32, 64, (3, 3), 1, 1, false))
    mixed_3a_branch_0 = Chain(MaxPool((3, 3), stride = 2))
    mixed_3a_branch_1 = Chain(conv2d(64, 96, (3, 3), 0, 2, false))
    mixed_4a_branch_0 = Chain(conv2d(160, 64, (1, 1), 0, 1, false),
                              conv2d(64, 96, (3, 3), 0, 1, false))
    mixed_4a_branch_1 = Chain(conv2d(160, 64, (1, 1), 0, 1, false),
                              conv2d(64, 64, (1, 7), (0, 3), 1, false),
                              conv2d(64, 64, (7, 1), (3, 0), 1, false),
                              conv2d(64, 96, (3, 3), 0, 1, false))
    mixed_5a_branch_0 =  Chain(conv2d(192, 192, (3, 3), 0, 2, false))
    mixed_5a_branch_1 = Chain(MaxPool((3, 3), stride = 2))

    Chain(conv2d_1a_3x3, conv2d_2a_3x3, conv2d_2b_3x3, 
          Parallel(cat_channels, mixed_3a_branch_0, mixed_3a_branch_1),
          Parallel(cat_channels, mixed_4a_branch_0, mixed_4a_branch_1),
          Parallel(cat_channels, mixed_5a_branch_0, mixed_5a_branch_1))
end

function inception_a(inc)
    branch_0 = Chain(conv2d(inc, 96, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(inc, 64, (1, 1), 0, 1, false),
                     conv2d(64, 96, (3, 3), 1, 1, false))
    branch_2 = Chain(conv2d(inc, 64, (1, 1), 0, 1, false),
                     conv2d(64, 96, (3, 3), 1, 1, false),
                     conv2d(96, 96, (3, 3), 1, 1, false))
    branch_3 = Chain(MeanPool((3, 3), pad = 1, stride = (1, 1)),
                     conv2d(384, 96, (1, 1), 0, 1, false))
    
    Parallel(cat_channels, branch_0, branch_1, branch_2, branch_3)
end


function inception_b(inc)
    branch_0 = Chain(conv2d(inc, 384, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(inc, 192, (1, 1), 0, 1, false),
                     conv2d(192, 224, (1, 7), (0, 3), 1, false),
                     conv2d(224, 256, (7, 1), (3, 0), 1, false))
    branch_2 = Chain(conv2d(inc, 192, (1, 1), 0, 1, false),
                     conv2d(192, 192, (7, 1), (3, 0), 1, false),
                     conv2d(192, 224, (1, 7), (0, 3), 1, false),
                     conv2d(224, 224, (7, 1), (3, 0), 1, false),
                     conv2d(224, 256, (1, 7), (0, 3), 1, false))
    branch_3 = Chain(MeanPool((3, 3), pad = 1, stride = (1, 1)),
                     conv2d(inc, 128, (1, 1), 0, 1, false))
    
    Parallel(cat_channels, branch_0, branch_1, branch_2, branch_3)
end

function reduction_a(inc, k, l, m, n)
    branch_0 = Chain(conv2d(inc, n, (3, 3), 0, 2, false))
    branch_1 = Chain(conv2d(inc, k, (1, 1), 0, 1, false),
                     conv2d(k, l, (3, 3), 1, 1, false),
                     conv2d(l, m, (3, 3), 0, 2, false))
    branch_2 = Chain(MaxPool((3, 3), stride = 2))

    Parallel(cat_channels,
             branch_0, branch_1, branch_2)
end

function reduction_b(inc)
    branch_0 = Chain(conv2d(inc, 192, (1, 1), 0, 1, false),
                     conv2d(192, 192, (3, 3), 0, 2, false))
    branch_1 = Chain(conv2d(inc, 256, (1, 1), 0, 1, false),
                     conv2d(256, 256, (1, 7), (0, 3), 1, false),
                     conv2d(256, 320, (7, 1), (3, 0), 1, false),
                     conv2d(320, 320, (3, 3), 0, 2, false))
    branch_2 = Chain(MaxPool((3, 3), stride = 2))
    
    Parallel(cat_channels, branch_0, branch_1, branch_2)
end

function inception_c(inc)
    branch_0 = Chain(conv2d(inc, 256, (1, 1), 0, 1, false))
    branch_1 = Chain(conv2d(inc, 384, (1, 1), 0, 1, false))
    branch_1_1 = Chain(conv2d(384, 256, (1, 3), (0, 1), 1, false))
    branch_1_2 = Chain(conv2d(384, 256, (3, 1), (1, 0), 1, false))
    branch_2 = Chain(conv2d(inc, 384, (1, 1), 0, 1, false),
                     conv2d(384, 448, (3, 1), (1, 0), 1, false),
                     conv2d(448, 512, (1, 3), (0, 1), 1, false))
    branch_2_1 = Chain(conv2d(512, 256, (1, 3), (0, 1), 1, false))
    branch_2_2 = Chain(conv2d(512, 256, (3, 1), (1, 0), 1, false))
    branch_3 = Chain(MeanPool((3, 3), pad = 1,stride = (1 ,1)),
                     conv2d(inc, 256, (1, 1), 0, 1, false))
    
    Parallel(cat_channels, branch_0, 
             Chain(branch_1, Parallel(cat_channels, branch_1_1, branch_1_2)), 
             Chain(branch_2,Parallel(cat_channels, branch_2_1, branch_2_2)), branch_3)
end

function inceptionv4(inc = 3, classes = 1000, k = 192, l = 224, m = 256, n = 384)
    Chain(stem(inc),
          inception_a(384),
          inception_a(384),
          inception_a(384),
          inception_a(384),
          reduction_a(384, k, l, m, n),
          inception_b(1024),
          inception_b(1024),
          inception_b(1024),
          inception_b(1024),
          inception_b(1024),
          inception_b(1024),
          inception_b(1024),
          reduction_b(1024),
          inception_c(1536),
          inception_c(1536),
          inception_c(1536),
          AdaptiveMeanPool((1, 1)),
          Flux.flatten,
          Dense(1536, classes)
          )
end