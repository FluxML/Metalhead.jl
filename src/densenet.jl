function bottleneck(inplanes, growth_rate)
    inner_channels = 4 * growth_rate
    m = Chain(conv_bn((1, 1), inplanes, inner_channels; usebias=false, rev=true)...,
              conv_bn((3, 3), inner_channels, growth_rate; pad=1, usebias=false, rev=true)...)

    SkipConnection(m, (mx, x) -> cat(x, mx; dims=3))
end

transition(inplanes, outplanes) = (conv_bn((1, 1), inplanes, outplanes; usebias=false, rev=true)...,
                                   MeanPool((2, 2)))

function make_dense_block(inplanes, growth_rate, nblock)
    layers = []
    for i in 1:nblock
        push!(layers, bottleneck(inplanes, growth_rate))
        inplanes += growth_rate
    end
    return layers
end

function densenet(nblocks=(6, 12, 24, 16); growth_rate=32, reduction=0.5, num_classes=1000)
    num_planes = 2 * growth_rate
    layers = []
    append!(layers, conv_bn((7, 7), 3, num_planes; stride=2, pad=(3, 3), usebias=false))
    push!(layers, MaxPool((3, 3), stride=2, pad=(1, 1)))

    for i in 1:3
        append!(layers, make_dense_block(num_planes, growth_rate, nblocks[i]))
        num_planes += nblocks[i] * growth_rate
        out_planes = Int(floor(num_planes * reduction))
        append!(layers, transition(num_planes, out_planes))
        num_planes = out_planes
    end

    append!(layers, make_dense_block(num_planes, growth_rate, nblocks[4]))
    num_planes += nblocks[4] * growth_rate
    push!(layers, BatchNorm(num_planes, relu))

    Chain(layers...,
          AdaptiveMeanPool((1, 1)),
          flatten,
          Dense(num_planes, num_classes))
end

densenet_121() = densenet()

densenet_161() = densenet((6, 12, 36, 24); growth_rate=64)

densenet_169() = densenet((6, 12, 32, 32))

densenet_201() = densenet((6, 12, 48, 32))