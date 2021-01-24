function inception_a(inplanes, pool_proj)
  branch1x1 = Chain(conv_bn((1, 1), inplanes, 64)...)
  
  branch5x5 = Chain(conv_bn((1, 1), inplanes, 48)...,
                    conv_bn((5, 5), 48, 64; pad=2)...)

  branch3x3 = Chain(conv_bn((1, 1), inplanes, 64)..., 
                    conv_bn((3, 3), 64, 96; pad=1)...,
                    conv_bn((3, 3), 96, 96; pad=1)...)

  branch_pool = Chain(MeanPool((3, 3), pad=1, stride=1),
                      conv_bn((1, 1), inplanes, pool_proj)...)

  return Parallel(cat_channels,
                  branch1x1, branch5x5, branch3x3, branch_pool)
end

function inception_b(inplanes)
  branch3x3_1 = Chain(conv_bn((3, 3), inplanes, 384; stride=2)...)

  branch3x3_2 = Chain(conv_bn((1, 1), inplanes, 64)...,
                      conv_bn((3, 3), 64, 96; pad=1)...,
                      conv_bn((3, 3), 96, 96; stride=2)...)

  branch_pool = Chain(MaxPool((3, 3), stride=2))

  return Parallel(cat_channels,
                  branch3x3_1, branch3x3_2, branch_pool)
end

function inception_c(inplanes, c7)
  branch1x1 = Chain(conv_bn((1, 1), inplanes, 192)...)

  branch7x7_1 = Chain(conv_bn((1, 1), inplanes, c7)...,
                      conv_bn((1, 7), c7, c7; pad=(0, 3))...,
                      conv_bn((7, 1), c7, 192; pad=(3, 0))...)

  branch7x7_2 = Chain(conv_bn((1, 1), inplanes, c7)...,
                      conv_bn((7, 1), c7, c7; pad=(3, 0))...,
                      conv_bn((1, 7), c7, c7; pad=(0, 3))...,
                      conv_bn((7, 1), c7, c7; pad=(3, 0))...,
                      conv_bn((1, 7), c7, 192; pad=(0, 3))...)

  branch_pool = Chain(MeanPool((3, 3), pad=1, stride=1), 
                      conv_bn((1, 1), inplanes, 192)...)

  return Parallel(cat_channels,
                  branch1x1, branch7x7_1, branch7x7_2, branch_pool)
end

function inception_d(inplanes)
  branch3x3 = Chain(conv_bn((1, 1), inplanes, 192)...,
                    conv_bn((3, 3), 192, 320; stride=2)...)

  branch7x7x3 = Chain(conv_bn((1, 1), inplanes, 192)...,
                      conv_bn((1, 7), 192, 192; pad=(0, 3))...,
                      conv_bn((7, 1), 192, 192; pad=(3, 0))...,
                      conv_bn((3, 3), 192, 192; stride=2)...)

  branch_pool = Chain(MaxPool((3, 3), stride=2))

  return Parallel(cat_channels,
                  branch3x3, branch7x7x3, branch_pool)
end

function inception_e(inplanes)
  branch1x1 = Chain(conv_bn((1, 1), inplanes, 320)...)

  branch3x3_1 = Chain(conv_bn((1, 1), inplanes, 384)...)
  branch3x3_1a = Chain(conv_bn((1, 3), 384, 384; pad=(0, 1))...)
  branch3x3_1b = Chain(conv_bn((3, 1), 384, 384; pad=(1, 0))...)

  branch3x3_2 = Chain(conv_bn((1, 1), inplanes, 448)...,
                      conv_bn((3, 3), 448, 384; pad=1)...)
  branch3x3_2a = Chain(conv_bn((1, 3), 384, 384; pad=(0, 1))...)
  branch3x3_2b = Chain(conv_bn((3, 1), 384, 384; pad=(1, 0))...)

  branch_pool = Chain(MeanPool((3, 3), pad=1, stride=1),
                      conv_bn((1, 1), inplanes, 192)...)

  return Parallel(cat_channels,
                  branch1x1,
                  Chain(branch3x3_1,
                        Parallel(cat_channels,
                                  branch3x3_1a, branch3x3_1b)),
      
                  Chain(branch3x3_2,
                        Parallel(cat_channels,
                                  branch3x3_2a, branch3x3_2b)),
                  branch_pool)
end

function inception3()
  layer = Chain(conv_bn((3, 3), 3, 32; stride=2)...,
                conv_bn((3, 3), 32, 32)...,
                conv_bn((3, 3), 32, 64; pad=1)...,
                MaxPool((3, 3), stride=2),
                conv_bn((1, 1), 64, 80)...,
                conv_bn((3, 3), 80, 192)...,
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

  return layer
end
