"""
    Bottle2Neck(inplanes, planes; expansion = 4, stride = 1, downsample = nothing,
                cardinality = 1, base_width = 26, scale = 4, stype = :normal)

Creates a bottle2neck block as defined in the paper for Res2Net.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments:
- `inplanes`: number of input channels
- `planes`: number of output channels
- `expansion`: expansion factor of the block
- `stride`: stride of the 3x3 convolution layer in the block
- `downsample`: if not nothing, downsample the input using the layer provided
- `cardinality`: number of convolution groups in the block (used for Res2NeXt)
- `base_width`: base width of the 3x3 convolution layer in the block
- `scale`: scale factor to control the scale dimension of the block
- `stype`: type of the block - either :normal or :stage. :normal is the default, 
           :stage is used for the first block of a new stage.
"""
struct Bottle2Neck
  scale
  stype
  width
  nums
  layers
  pool
  downsample
end

function Bottle2Neck(inplanes, planes; expansion = 4, stride = 1, downsample = nothing,
                     cardinality = 1, base_width = 26, scale = 4, stype = :normal)
  @assert stype in [:normal, :stage]
  width = floor(Int, planes * base_width / 64) * cardinality
  nums = (scale == 1) ? 1 : scale - 1
  layers = Chain(conv_bn((1, 1), inplanes, width * scale; groups = cardinality, bias = false),
                 [conv_bn((3, 3), width, width; groups = cardinality, stride, pad = 1, 
                    bias = false) for _ in 1:nums]...,
                 conv_bn((1, 1), width * scale, planes * expansion; bias = false))
  pool = (stype == :stage) ? MeanPool((3, 3); stride, pad = 1) : nothing
  Bottle2Neck(scale, stype, width, nums, layers, pool, downsample)
end

@functor Bottle2Neck

function (m::Bottle2Neck)(x)
  residual = x
  out = m.layers[1](x)
  spx = [out[:, :, i:min(i + m.width - 1, end), :] for i in 1:m.width]
  local sp
  for i in 1:m.nums
    sp = (i == 1 || m.stype == :stage) ? spx[i] : sp + spx[i]
    sp = m.layers[i + 1](sp)
    out = (i == 1) ? sp : cat(out, sp; dims = 3)
  end
  if m.scale != 1 && m.stype == :normal
    out = cat(out, spx[m.nums]; dims = 3)
  elseif m.scale != 1 && m.stype == :stage
    out = cat(out, m.pool(spx[m.nums]); dims = 3)
  end
  out = m.layers[end](out)
  if !isnothing(m.downsample)
    residual = m.downsample(residual)
  end
  return relu.(out + residual)
end

"""
    res2net(block_config; cardinality = 1, expansion = 4, base_width = 26, scale = 4, 
            nclasses = 1000)

Creates the layers for a Res2Net model as defined in the paper.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments:
- `block_config`: list of layers in each block of the model
- `cardinality`: number of convolution groups in the block (used for Res2NeXt)
- `expansion`: expansion factor of the block
- `base_width`: base width of the 3x3 convolution layer in the block
- `scale`: scale factor to control the scale dimension of the block
- `nclasses`: number of output classes
"""
function res2net(block_config; cardinality = 1, expansion = 4, base_width = 26, scale = 4, 
                 nclasses = 1000)
  inplanes = 64
  base_inplanes = inplanes

  layers = []
  push!(layers, Chain(conv_bn((7, 7), 3, base_inplanes; stride = 2, pad = 3, bias = false)...,
                      MaxPool((3, 3); pad = 1, stride = 2)))

  for (i, nblocks) in enumerate(block_config)
    stride = (i == 1) ? 1 : 2
    planes = base_inplanes * (2 ^ (i - 1))
    downsample = (stride != 1 || inplanes != planes * expansion) ?
                 conv_bn((1, 1), inplanes, planes * expansion, identity; stride, bias = false) : 
                 nothing
    push!(layers, Bottle2Neck(inplanes, planes; stype = :stage, cardinality, expansion, stride, 
                              downsample, base_width, scale))
    inplanes = planes * expansion
    append!(layers, [Bottle2Neck(inplanes, planes; cardinality, expansion, base_width, scale) 
                      for _ in 2:nblocks])
  end

  head = Chain(GlobalMeanPool(), MLUtils.flatten, Dense(base_inplanes * 8 * expansion, nclasses))

  return Chain(Chain(layers...), head)
end

const depth_config = Dict(50 => (3, 4, 6, 3),
                          101 => (3, 4, 23, 3),
                          152 => (3, 8, 36, 3))

struct Res2Net
  layers
end

@functor Res2Net

(m::Res2Net)(x) = m.layers(x)

backbone(m::Res2Net) = m.layers[1]
classifier(m::Res2Net) = m.layer[2]

"""
    Res2Net(depth::Int = 50; nclasses = 1000)

Creates a Res2Net model as defined in the paper.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments:
- `depth`: depth of the network. One of (50, 101, 152)
- `nclasses`: number of output classes
"""
function Res2Net(depth::Int = 50; nclasses = 1000)
  @assert depth in keys(depth_config) "`config` must be one of $(sort(collect(keys(depth_config))))"
  layers = res2net(depth_config[depth]; nclasses)

  Res2Net(layers)
end

"""
    Res2NeXt(depth::Int = 50; cardinality = 8, nclasses = 1000)

Creates a Res2NeXt model as defined in the paper.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments:
- `depth`: depth of the network. One of (50, 101, 152)
- `cardinality`: number of convolution groups in the 3x3 convolution layer in each block
- `nclasses`: number of output classes
"""
function Res2NeXt(depth::Int = 50; cardinality = 8, nclasses = 1000)
  @assert depth in keys(depth_config) "`config` must be one of $(sort(collect(keys(depth_config))))"
  layers = res2net(depth_config[depth]; cardinality, nclasses)

  Res2Net(layers)
end
