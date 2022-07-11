"""
    bottle2neck(inplanes, planes; stride = 1, downsample = identity,
                cardinality = 1, base_width = 26, scale = 4, dilation = 1,
                activation = relu, norm_layer = BatchNorm,
                attn_fn = planes -> identity, kwargs...)

Creates a bottleneck block as described in the Res2Net paper.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `inplanes`: number of input feature maps
  - `planes`: number of feature maps for the block
  - `stride`: the stride of the block
  - `downsample`: the downsampling function to use
  - `cardinality`: the number of groups in the 3x3 convolutions.
  - `base_width`: the number of output feature maps for each convolutional group.
  - `scale`: the number of feature groups in the block. See the
    [paper](https://arxiv.org/abs/1904.01169) for more details.
  - `first_dilation`: the dilation of the 3x3 convolution.
  - `activation`: the activation function to use.
  - `connection`: the function applied to the output of residual and skip paths in
    a block. See [`addact`](#) and [`actadd`](#) for an example. Note that this uses
    [PartialFunctions.jl](https://github.com/archermarx/PartialFunctions.jl) to pass
    in the activation function with the notation `addact\$activation`.
  - `norm_layer`: the normalization layer to use.
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
  - `attn_args`: a NamedTuple that contains none, some or all of the arguments to be passed to the
    attention function.
"""
function bottle2neck(inplanes, planes; stride = 1, downsample = identity,
                     cardinality = 1, base_width = 26, scale = 4, first_dilation = 1,
                     activation = relu, norm_layer = BatchNorm,
                     attn_fn = planes -> identity, kwargs...)
    expansion = expansion_factor(bottle2neck)
    width = fld(planes * base_width, 64) * cardinality
    outplanes = planes * expansion
    is_first = stride > 1 || downsample !== identity
    pool = is_first && scale > 1 ? MeanPool((3, 3); stride, pad = 1) : identity
    conv_bns = [Chain(Conv((3, 3), width => width; stride, pad = first_dilation,
                           dilation = first_dilation, groups = cardinality, bias = false),
                      norm_layer(width, activation))
                for _ in 1:(max(1, scale - 1))]
    reslayer = is_first ? Parallel(cat_channels, pool, conv_bns...) :
               Parallel(cat_channels, identity, PairwiseFusion(+, conv_bns...))
    tuplify(x) = is_first ? tuple(x...) : tuple(x[1], tuple(x[2:end]...))
    return Chain(Parallel(+, downsample,
                          Chain(Conv((1, 1), inplanes => width * scale; bias = false),
                                norm_layer(width * scale, activation),
                                x -> chunk(x; size = width, dims = 3),
                                tuplify,
                                reslayer,
                                Conv((1, 1), width * scale => outplanes; bias = false),
                                norm_layer(outplanes, activation),
                                attn_fn(outplanes))), relu)
end
expansion_factor(::typeof(bottle2neck)) = 4

"""
    Res2Net(depth::Integer; pretrain = false, scale = 4, base_width = 26,
            inchannels = 3, nclasses = 1000)

Creates a Res2Net model with the specified depth, scale, and base width.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the Res2Net model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `scale`: the number of feature groups in the block. See the
    [paper](https://arxiv.org/abs/1904.01169) for more details.
  - `base_width`: the number of feature maps in each group.
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes
"""
struct Res2Net
    layers::Any
end
@functor Res2Net

(m::Res2Net)(x) = m.layers(x)

function Res2Net(depth::Integer; pretrain = false, scale = 4, base_width = 26,
                 inchannels = 3, nclasses = 1000)
    @assert depth in [50, 101, 152]
    "Invalid depth. Must be one of [50, 101, 152]"
    layers = resnet(bottle2neck, resnet_config[depth][2], :C; inchannels, nclasses,
                    block_args = (; scale, base_width))
    if pretrain
        loadpretrain!(layers, string("resnet", depth))
    end
    return Res2Net(layers)
end

"""
    Res2NeXt(depth::Integer; pretrain = false, scale = 4, base_width = 4, cardinality = 8,
             inchannels = 3, nclasses = 1000)

Creates a Res2NeXt model with the specified depth, scale, base width and cardinality.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `depth`: one of `[50, 101, 152]`. The depth of the Res2Net model.
  - `pretrain`: set to `true` to load the model with pre-trained weights for ImageNet
  - `scale`: the number of feature groups in the block. See the
    [paper](https://arxiv.org/abs/1904.01169) for more details.
  - `base_width`: the number of feature maps in each group.
  - `cardinality`: the number of groups in the 3x3 convolutions.
  - `inchannels`: the number of input channels.
  - `nclasses`: the number of output classes
"""
struct Res2NeXt
    layers::Any
end
@functor Res2NeXt

(m::Res2NeXt)(x) = m.layers(x)

function Res2NeXt(depth::Integer; pretrain = false, scale = 4, base_width = 4,
                  cardinality = 8,
                  inchannels = 3, nclasses = 1000)
    @assert depth in [50, 101, 152]
    "Invalid depth. Must be one of [50, 101, 152]"
    layers = resnet(bottle2neck, resnet_config[depth][2], :C; inchannels, nclasses,
                    block_args = (; scale, base_width, cardinality))
    if pretrain
        loadpretrain!(layers, string("resnext", depth, "_", cardinality, "x", base_width))
    end
    return Res2NeXt(layers)
end

backbone(m::Res2NeXt) = m.layers[1]
classifier(m::Res2NeXt) = m.layers[2]
