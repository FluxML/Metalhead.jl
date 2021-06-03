"""
    inceptionblock(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_3x3, pool_proj)

Create an inception module for use in GoogLeNet
([reference](https://arxiv.org/abs/1409.4842v1)).

# Arguments
- `inplanes`: the number of input feature maps
- `out_1x1`: the number of output feature maps for the 1x1 convolution (branch 1)
- `red_3x3`: the number of output feature maps for the 3x3 reduction convolution (branch 2)
- `out_3x3`: the number of output feature maps for the 3x3 convolution (branch 2)
- `red_5x5`: the number of output feature maps for the 5x5 reduction convolution (branch 3)
- `out_5x5`: the number of output feature maps for the 5x5 convolution (branch 3)
- `pool_proj`: the number of output feature maps for the pooling projection (branch 4)
"""
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

"""
    googlenet()

Create an Inception-v1 model (commonly referred to as GoogLeNet)
([reference](https://arxiv.org/abs/1409.4842v1)).
"""
function googlenet()
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

  return layers
end

"""
    GoogLeNet(; pretrain=false)

Create an Inception-v1 model (commonly referred to as `GoogLeNet`)
([reference](https://arxiv.org/abs/1409.4842v1)).
Set `pretrain=true` to load the model with pre-trained weights for ImageNet.

See also [`googlenet`](#).
"""
struct GoogLeNet{T}
  layers::T

  function GoogLeNet(; pretrain=false)
    layers = googlenet()

    pretrain && Flux.loadparams!(layers, weights("googlenet"))
    new{typeof(layers)}(layers)
  end
end

(m::GoogLeNet)(x) = m.layers(x)
