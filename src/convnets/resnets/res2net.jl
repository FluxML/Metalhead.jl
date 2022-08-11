"""
    bottle2neck(inplanes::Integer, planes::Integer; stride::Integer = 1,
                cardinality::Integer = 1, base_width::Integer = 26,
                scale::Integer = 4, activation = relu, norm_layer = BatchNorm,
                revnorm::Bool = false, attn_fn = planes -> identity)

Creates a bottleneck block as described in the Res2Net paper.
([reference](https://arxiv.org/abs/1904.01169))

# Arguments

  - `inplanes`: number of input feature maps
  - `planes`: number of feature maps for the block
  - `stride`: the stride of the block
  - `cardinality`: the number of groups in the 3x3 convolutions.
  - `base_width`: the number of output feature maps for each convolutional group.
  - `scale`: the number of feature groups in the block. See the [paper](https://arxiv.org/abs/1904.01169)
    for more details.
  - `activation`: the activation function to use.
  - `norm_layer`: the normalization layer to use.
  - `revnorm`: set to `true` to place the batch norm before the convolution
  - `attn_fn`: the attention function to use. See [`squeeze_excite`](#) for an example.
"""
function bottle2neck(inplanes::Integer, planes::Integer; stride::Integer = 1,
                     cardinality::Integer = 1, base_width::Integer = 26,
                     scale::Integer = 4, activation = relu, is_first::Bool = false,
                     norm_layer = BatchNorm, revnorm::Bool = false,
                     attn_fn = planes -> identity)
    width = fld(planes * base_width, 64) * cardinality
    outplanes = planes * 4
    pool = is_first && scale > 1 ? MeanPool((3, 3); stride, pad = 1) : identity
    conv_bns = [Chain(conv_norm((3, 3), width => width, activation; norm_layer, stride,
                                pad = 1, groups = cardinality, bias = false)...)
                for _ in 1:max(1, scale - 1)]
    reslayer = is_first ? Parallel(cat_channels, pool, conv_bns...) :
               Parallel(cat_channels, identity, Chain(PairwiseFusion(+, conv_bns...)))
    tuplify = is_first ? x -> tuple(x...) : x -> tuple(x[1], tuple(x[2:end]...))
    layers = [
        conv_norm((1, 1), inplanes => width * scale, activation;
                  norm_layer, revnorm, bias = false)...,
        chunk$(; size = width, dims = 3), tuplify, reslayer,
        conv_norm((1, 1), width * scale => outplanes, activation;
                  norm_layer, revnorm, bias = false)...,
        attn_fn(outplanes),
    ]
    return Chain(filter(!=(identity), layers)...)
end

function bottle2neck_builder(block_repeats::AbstractVector{<:Integer};
                             inplanes::Integer = 64, cardinality::Integer = 1,
                             base_width::Integer = 26, scale::Integer = 4,
                             expansion::Integer = 4, norm_layer = BatchNorm,
                             revnorm::Bool = false, activation = relu,
                             attn_fn = planes -> identity,
                             stride_fn = resnet_stride, planes_fn = resnet_planes,
                             downsample_tuple = (downsample_conv, downsample_identity))
    planes_vec = collect(planes_fn(block_repeats))
    # closure over `idxs`
    function get_layers(stage_idx::Integer, block_idx::Integer)
        # This is needed for block `inplanes` and `planes` calculations
        schedule_idx = sum(block_repeats[1:(stage_idx - 1)]) + block_idx
        planes = planes_vec[schedule_idx]
        inplanes = schedule_idx == 1 ? inplanes : planes_vec[schedule_idx - 1] * expansion
        # `resnet_stride` is a callback that the user can tweak to change the stride of the
        # blocks. It defaults to the standard behaviour as in the paper
        stride = stride_fn(stage_idx, block_idx)
        downsample_fn = (stride != 1 || inplanes != planes * expansion) ?
                        downsample_tuple[1] : downsample_tuple[2]
        is_first = (stride > 1 || downsample_fn != downsample_tuple[2]) ? true : false
        block = bottle2neck(inplanes, planes; stride, cardinality, base_width, scale,
                            activation, is_first, norm_layer, revnorm, attn_fn)
        downsample = downsample_fn(inplanes, planes * expansion; stride, norm_layer,
                                   revnorm)
        return block, downsample
    end
    return get_layers
end

"""
    Res2Net(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
            base_width::Integer = 26, inchannels::Integer = 3,
            nclasses::Integer = 1000)

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

function Res2Net(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
                 base_width::Integer = 26, inchannels::Integer = 3,
                 nclasses::Integer = 1000)
    _checkconfig(depth, keys(LRESNET_CONFIGS))
    layers = resnet(bottle2neck, LRESNET_CONFIGS[depth][2]; base_width, scale,
                    inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("Res2Net", depth, "_", base_width, "x", scale))
    end
    return Res2Net(layers)
end

(m::Res2Net)(x) = m.layers(x)

backbone(m::Res2Net) = m.layers[1]
classifier(m::Res2Net) = m.layers[2]

"""
    Res2NeXt(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
             base_width::Integer = 4, cardinality::Integer = 8,
             inchannels::Integer = 3, nclasses::Integer = 1000)

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

function Res2NeXt(depth::Integer; pretrain::Bool = false, scale::Integer = 4,
                  base_width::Integer = 4, cardinality::Integer = 8,
                  inchannels::Integer = 3, nclasses::Integer = 1000)
    _checkconfig(depth, keys(LRESNET_CONFIGS))
    layers = resnet(bottle2neck, LRESNET_CONFIGS[depth][2]; base_width, scale,
                    cardinality, inchannels, nclasses)
    if pretrain
        loadpretrain!(layers,
                      string("Res2NeXt", depth, "_", base_width, "x", cardinality,
                             "x", scale))
    end
    return Res2NeXt(layers)
end

(m::Res2NeXt)(x) = m.layers(x)

backbone(m::Res2NeXt) = m.layers[1]
classifier(m::Res2NeXt) = m.layers[2]
