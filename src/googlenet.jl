function conv_block(kernelsize::Tuple{Int64,Int64}, inplanes::Int64, outplanes::Int64; stride::Int64=1, pad::Int64=0)
    conv_layer = []
    push!(conv_layer, Conv(kernelsize, inplanes => outplanes, stride = stride, pad = pad)) 
    push!(conv_layer, BatchNorm(outplanes, relu))
    return conv_layer
  end
  
  function inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj)
    branch1 = Chain(conv_block((1,1), inplanes, out_1x1)...)
  
    branch2 = Chain(conv_block((1,1), inplanes, red_3x3)...,
               conv_block((3,3), red_3x3, out_3x3; pad=1)...)        
  
    branch3 = Chain(conv_block((1,1), inplanes, red_5x5)...,
               conv_block((5,5), red_5x5, out_5x5; pad=2)...) 
  
    branch4 = Chain(MaxPool((3, 3), stride=1, pad=1),
               conv_block((1,1), inplanes, pool_proj)...)
  
    inception_layer = x -> begin
      y1 = branch1(x)
      y2 = branch2(x)
      y3 = branch3(x)
      y4 = branch4(x)
    
      return cat(y1, y2, y3, y4; dims=3)
  
    end
  
  end
  
  function googlenet()
    layers = Chain(conv_block((7,7), 3, 64; stride=2, pad=3)...,
                   MaxPool((3,3), stride=2, pad=1),
                   conv_block((1,1), 64, 64)...,
                   conv_block((3,3), 64, 192; pad=1)...,
                   MaxPool((3,3), stride=2, pad=1),
                   inceptionblock(192, 64, 96, 128, 16, 32, 32),
                   inceptionblock(256, 128, 128, 192, 32, 96, 64),
                   MaxPool((3,3), stride=2, pad=1),
                   inceptionblock(480, 192, 96, 208, 16, 48, 64),
                   inceptionblock(512, 160, 112, 224, 24, 64, 64),
                   inceptionblock(512, 128, 128, 256, 24, 64, 64),
                   inceptionblock(512, 112, 144, 288, 32, 64, 64),
                   inceptionblock(528, 256, 160, 320, 32, 128, 128),
                   MaxPool((3,3), stride=2, pad=1),
                   inceptionblock(832, 256, 160, 320, 32, 128, 128),
                   inceptionblock(832, 384, 192, 384, 48, 128, 128),
                   AdaptiveMeanPool((1,1)),
                   flatten,
                   Dropout(0.2),
                   Dense(1024, 1000), softmax)
    Flux.testmode!(layers, false)
    return layers
  end  