function conv_block(kernelsize::Tuple{Int64,Int64}, inplanes::Int64, outplanes::Int64; stride::Int64=1, pad::Union{Int64,Tuple{Int64,Int64}}=0)
    conv_layer = []
    push!(conv_layer, Conv(kernelsize, inplanes => outplanes, stride=stride, pad=pad))
    push!(conv_layer, BatchNorm(outplanes, relu))
    return conv_layer
end

function inception_a(inplanes, pool_proj)
    branch1x1 = Chain(conv_block((1, 1), inplanes, 64)...)
    
    branch5x5 = Chain(conv_block((1, 1), inplanes, 48)...,
                      conv_block((5, 5), 48, 64; pad=2)...)

    branch3x3 = Chain(conv_block((1, 1), inplanes, 64)..., 
                      conv_block((3, 3), 64, 96; pad=1)...,
                      conv_block((3, 3), 96, 96; pad=1)...)

    branch_pool = Chain(MeanPool((3, 3), pad=1, stride=1),
                        conv_block((1, 1), inplanes, pool_proj)...)

    layer = x -> begin 
        y1 = branch1x1(x)
        y2 = branch5x5(x)
        y3 = branch3x3(x)
        y4 = branch_pool(x)
        
        return cat(y1, y2, y3, y4; dims=3)
    end
end

function inception_b(inplanes)
    branch3x3_1 = Chain(conv_block((3, 3), inplanes, 384; stride=2)...)

    branch3x3_2 = Chain(conv_block((1, 1), inplanes, 64)...,
                        conv_block((3, 3), 64, 96; pad=1)...,
                        conv_block((3, 3), 96, 96; stride=2)...)

    branch_pool = Chain(MaxPool((3, 3), stride=2))

    layer = x -> begin
        y1 = branch3x3_1(x)
        y2 = branch3x3_2(x)
        y3 = branch_pool(x)

        return cat(y1, y2, y3; dims=3)
    end
end

function inception_c(inplanes, c7)
    branch1x1 = Chain(conv_block((1, 1), inplanes, 192)...)

    branch7x7_1 = Chain(conv_block((1, 1), inplanes, c7)...,
                        conv_block((1, 7), c7, c7; pad=(0, 3))...,
                        conv_block((7, 1), c7, 192; pad=(3, 0))...)

    branch7x7_2 = Chain(conv_block((1, 1), inplanes, c7)...,
                        conv_block((7, 1), c7, c7; pad=(3, 0))...,
                        conv_block((1, 7), c7, c7; pad=(0, 3))...,
                        conv_block((7, 1), c7, c7; pad=(3, 0))...,
                        conv_block((1, 7), c7, 192; pad=(0, 3))...)

    branch_pool = Chain(MeanPool((3, 3), pad=1, stride=1), 
                        conv_block((1, 1), inplanes, 192)...)

    layer = x -> begin
        y1 = branch1x1(x)
        y2 = branch7x7_1(x)
        y3 = branch7x7_2(x)
        y4 = branch_pool(x)

        return cat(y1, y2, y3, y4; dims=3)
    end
end

function inception_d(inplanes)
    branch3x3 = Chain(conv_block((1, 1), inplanes, 192)...,
                      conv_block((3, 3), 192, 320; stride=2)...)

    branch7x7x3 = Chain(conv_block((1, 1), inplanes, 192)...,
                        conv_block((1, 7), 192, 192; pad=(0, 3))...,
                        conv_block((7, 1), 192, 192; pad=(3, 0))...,
                        conv_block((3, 3), 192, 192; stride=2)...)

    branch_pool = Chain(MaxPool((3, 3), stride=2))

    layer = x -> begin
        y1 = branch3x3(x)
        y2 = branch7x7x3(x)
        y3 = branch_pool(x)

        return cat(y1, y2, y3; dims=3)
    end
end

function inception_e(inplanes)
    branch1x1 = Chain(conv_block((1, 1), inplanes, 320)...)

    branch3x3_1 = Chain(conv_block((1, 1), inplanes, 384)...)
    branch3x3_1a = Chain(conv_block((1, 3), 384, 384; pad=(0, 1))...)
    branch3x3_1b = Chain(conv_block((3, 1), 384, 384; pad=(1, 0))...)

    branch3x3_2 = Chain(conv_block((1, 1), inplanes, 448)...,
                        conv_block((3, 3), 448, 384; pad=1)...)
    branch3x3_2a = Chain(conv_block((1, 3), 384, 384; pad=(0, 1))...)
    branch3x3_2b = Chain(conv_block((3, 1), 384, 384; pad=(1, 0))...)

    branch_pool = Chain(MeanPool((3, 3), pad=1, stride=1),
                        conv_block((1, 1), inplanes, 192)...)

    layer = x -> begin
        y1 = branch1x1(x)
        
        y2 = branch3x3_1(x)
        y2 = cat(branch3x3_1a(y2), branch3x3_1b(y2); dims=3)
        
        y3 = branch3x3_2(x)
        y3 = cat(branch3x3_2a(y3), branch3x3_2b(y3); dims=3)
        
        y4 = branch_pool(x)

        return cat(y1, y2, y3, y4; dims=3)
    end
end

function inception3()
    layer = Chain(conv_block((3, 3), 3, 32; stride=2)...,
                  conv_block((3, 3), 32, 32)...,
                  conv_block((3, 3), 32, 64; pad=1)...,
                  MaxPool((3, 3), stride=2),
                  conv_block((1, 1), 64, 80)...,
                  conv_block((3, 3), 80, 192)...,
                  MaxPool((3, 3), stride=2),
                  inception_a(192, 32),
                  inception_a(256, 64),
                  inception_a(288, 64),
                  inception_b(288),
                  inception_c(768, 128),
                  inception_c(768, 160),
                  inception_c(768, 160),
                  inception_c(768, 192),
                  inception_d(768),
                  inception_e(1280),
                  inception_e(2048),
                  AdaptiveMeanPool((1, 1)),
                  Dropout(0.2),
                  flatten,
                  Dense(2048, 1000))
    Flux.testmode!(layer, false)
    return layer
end
