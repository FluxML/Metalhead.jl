function fire(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)
  branch_1 = Conv((1, 1), inplanes => squeeze_planes, relu)
  branch_2 = Conv((1, 1), squeeze_planes => expand1x1_planes, relu)
  branch_3 = Conv((3, 3), squeeze_planes => expand3x3_planes, pad=1, relu)

  return Chain(branch_1,
               Parallel(cat_channels,
                        branch_2,
                        branch_3))
end

function squeezenet(; pretrain=false)
  layers = Chain(Conv((3, 3), 3 => 64, relu, stride=2),
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
                 flatten)

  pretrain && Flux.loadparams!(layers, weights("squeezenet"))
  return layers
end