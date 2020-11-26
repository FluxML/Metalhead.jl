function fire(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)
    branch_1 = Conv((1, 1), inplanes => squeeze_planes, relu)
    branch_2 = Conv((1, 1), squeeze_planes => expand1x1_planes, relu)
    branch_3 = Conv((3, 3), squeeze_planes => expand3x3_planes, pad=1, relu)

    layer = x -> begin
        y1 = branch_1(x)
        y2 = branch_2(y1)
        y3 = branch_3(y1)

        return cat(y2, y3; dims=3)
    end
end

function squeezenet()
    layer = Chain(Conv((3, 3), 3 => 64, relu, stride=2),
                   MaxPool((3, 3), stride=2),
                   fire(64, 16, 64, 64),
                   fire(128, 16, 64, 64),
                   MaxPool((3, 3), stride=2),
                   fire(128, 32, 128, 128),
                   fire(256, 32, 128, 128),
                   MaxPool((3, 3), stride=2),
                   fire(256, 48, 192, 192),
                   fire(384, 48, 192, 192),
                   fire(384, 64, 256, 256),
                   fire(512, 64, 256, 256),
                   Dropout(0.5),
                   Conv((1, 1), 512 => 1000, relu),
                   AdaptiveMeanPool((1, 1)),
                   flatten, softmax)
    Flux.testmode!(layer, false)
    return layer
end