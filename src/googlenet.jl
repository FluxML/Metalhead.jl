function inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj)
    branch1 = Chain(Conv((1, 1), inplanes => out_1x1))
  
    branch2 = Chain(Conv((1, 1), inplanes => red_3x3),
                    Conv((3, 3), red_3x3 => out_3x3; pad=1))        
  
    branch3 = Chain(Conv((1, 1), inplanes => red_5x5),
                    Conv((5, 5), red_5x5 => out_5x5; pad=2)) 
  
    branch4 = Chain(MaxPool((3, 3), stride=1, pad=1),
                    Conv((1, 1), inplanes => pool_proj))
  
    return Parallel(cat_channels,
                    branch1, branch2, branch3, branch4)
end
  
function googlenet(; pretrain=false)
  layers = Chain(Conv((7, 7), 3 => 64; stride=2, pad=3),
                 MaxPool((3, 3), stride=2, pad=1),
                 Conv((1, 1), 64 => 64),
                 Conv((3, 3), 64 => 192; pad=1),
                 MaxPool((3, 3), stride=2, pad=1),
                 inceptionblock(192, 64, 96, 128, 16, 32, 32),
                 inceptionblock(256, 128, 128, 192, 32, 96, 64),
                 MaxPool((3, 3), stride=2, pad=1),
                 inceptionblock(480, 192, 96, 208, 16, 48, 64),
                 inceptionblock(512, 160, 112, 224, 24, 64, 64),
                 inceptionblock(512, 128, 128, 256, 24, 64, 64),
                 inceptionblock(512, 112, 144, 288, 32, 64, 64),
                 inceptionblock(528, 256, 160, 320, 32, 128, 128),
                 MaxPool((3, 3), stride=2, pad=1),
                 inceptionblock(832, 256, 160, 320, 32, 128, 128),
                 inceptionblock(832, 384, 192, 384, 48, 128, 128),
                 AdaptiveMeanPool((1, 1)),
                 flatten,
                 Dropout(0.4),
                 Dense(1024, 1000))

  pretrain && Flux.loadparams!(layers, weights("googlenet"))
  return layers
end  