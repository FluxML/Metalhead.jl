basicblock(inplanes, outplanes, downsample = false) = downsample ?
  Chain(Conv((3, 3), inplanes => outplanes[1], stride = 2, pad = 1),
        BatchNorm(outplanes[1], relu),
        Conv((3, 3), outplanes[1] => outplanes[2], stride = 1, pad = 1),
        BatchNorm(outplanes[2], relu)) :
  Chain(Conv((3, 3), inplanes => outplanes[1], stride = 1, pad = 1),
        BatchNorm(outplanes[1], relu),
        Conv((3, 3), outplanes[1] => outplanes[2], stride = 1, pad = 1),
        BatchNorm(outplanes[2], relu))

bottleneck(inplanes, outplanes, downsample = false) = downsample ?
  Chain(Conv((1, 1), inplanes => outplanes[1], stride = 2),
        BatchNorm(outplanes[1], relu),
        Conv((3, 3), outplanes[1] => outplanes[2], stride = 1, pad = 1),
        BatchNorm(outplanes[2], relu),
        Conv((1, 1), outplanes[2] => outplanes[3], stride = 1),
        BatchNorm(outplanes[3], relu)) :
  Chain(Conv((1, 1), inplanes => outplanes[1], stride = 1),
        BatchNorm(outplanes[1], relu),
        Conv((3, 3), outplanes[1] => outplanes[2], stride = 1, pad = 1),
        BatchNorm(outplanes[2], relu),
        Conv((1, 1), outplanes[2] => outplanes[3], stride = 1),
        BatchNorm(outplanes[3], relu))

function projection(inplanes, outplanes, downsample = false)
  shortcut = downsample ? 
    Chain(Conv((1, 1), inplanes => outplanes, stride = 2),
          BatchNorm((outplanes), relu)) :
    Chain(Conv((1, 1), inplanes => outplanes, stride = 1),
          BatchNorm((outplanes), relu))
  return (x, y) -> x + shortcut(y)
end

# array -> PaddedView(0, array, outplanes) for zero padding arrays
function identity(inplanes, outplanes)
  if outplanes[end] > inplanes
    pool = MaxPool((1, 1), stride = 2)
    return (x, y) -> begin
      y = pool(y)
      y = cat(y, zeros(eltype(y), size(y, 1), size(y, 2), outplanes[end] - inplanes, size(y, 4)); dims = 3)
      x + y
    end
  else
    return +
  end
end

function resnet(block, shortcut_config, channel_config, block_config)
  inplanes = 64
  baseplanes = 64
  layers = []
  push!(layers, Conv((7, 7), 3=>inplanes, stride=(2, 2), pad=(3, 3)))
  push!(layers, BatchNorm(inplanes, relu))
  push!(layers, MaxPool((3, 3), stride=(2, 2), pad=(1, 1)))
  for (i, nrepeats) in enumerate(block_config)
    outplanes = baseplanes .* channel_config
    if shortcut_config == :A
      push!(layers, SkipConnection(block(inplanes, outplanes, i != 1),
                                   identity(inplanes, outplanes)))
    elseif shortcut_config == :B || shortcut_config == :C
      push!(layers, SkipConnection(block(inplanes, outplanes, i != 1),
                                   projection(inplanes, outplanes[end], i != 1)))
    end
    inplanes = outplanes[end]
    for j in 2:nrepeats
      if shortcut_config == :A || shortcut_config == :B
        push!(layers, SkipConnection(block(inplanes, outplanes, false),
                                     identity(inplanes, outplanes[end])))
      elseif shortcut_config == :C
        push!(layers, SkipConnection(block(inplanes, outplanes, false),
                                     projection(inplanes, outplanes, false)))
      end
      inplanes = outplanes[end]
    end
    baseplanes *= 2
  end
  push!(layers, AdaptiveMeanPool((1, 1)))
  push!(layers, flatten)
  push!(layers, Dense(inplanes, 1000))
  layers = Chain(layers...)
  Flux.testmode!(layers, false)
  return layers
end

const resnet_config =
Dict("resnet18" => ([1, 1], [2, 2, 2, 2]),
     "resnet34" => ([1, 1], [3, 4, 6, 3]),
     "resnet50" => ([1, 1, 4], [3, 4, 6, 3]),
     "resnet101" => ([1, 1, 4], [3, 4, 23, 3]),
     "resnet152" => ([1, 1, 4], [3, 8, 36, 3]))

resnet18() = resnet(basicblock, :A, resnet_config["resnet18"]...)

resnet34() = resnet(basicblock, :A, resnet_config["resnet34"]...)

resnet50() = resnet(bottleneck, :B, resnet_config["resnet50"]...)

resnet101() = resnet(bottleneck, :B, resnet_config["resnet101"]...)

resnet152() = resnet(bottleneck, :B, resnet_config["resnet152"]...)