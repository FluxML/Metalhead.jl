function bottle2neck(inplanes, planes; stride = 1, downsample = identity,
                     cardinality = 1, base_width = 26, scale = 4, dilation = 1,
                     activation = relu, norm_layer = BatchNorm,
                     attn_fn = planes -> identity, kwargs...)
    expansion = expansion_factor(bottle2neck)
    width = fld(planes * base_width, 64) * cardinality
    outplanes = planes * expansion
    is_first = stride > 1 || downsample !== identity
    pool = is_first && scale > 1 ? MeanPool((3, 3); stride, pad = 1) : identity
    conv_bns = [Chain(Conv((3, 3), width => width; stride, pad = dilation, dilation,
                           groups = cardinality, bias = false),
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

struct Res2Net
    layers::Any
end
@functor Res2Net

(m::Res2Net)(x) = m.layers(x)

function Res2Net(depth::Integer; pretrain = false, inchannels = 3, nclasses = 1000)
    @assert depth in [18, 34, 50, 101, 152]
    "Invalid depth. Must be one of [18, 34, 50, 101, 152]"
    layers = resnet(bottle2neck, resnet_config[depth][2], :C; inchannels, nclasses)
    if pretrain
        loadpretrain!(layers, string("resnet", depth))
    end
    return Res2Net(layers)
end

struct Res2NeXt
    layers::Any
end
@functor Res2NeXt

(m::Res2NeXt)(x) = m.layers(x)

function Res2NeXt(depth::Integer; pretrain = false, cardinality = 32, base_width = 4,
                  inchannels = 3, nclasses = 1000)
    @assert depth in [50, 101, 152]
    "Invalid depth. Must be one of [50, 101, 152]"
    layers = resnet(bottle2neck, resnet_config[depth][2], :C; inchannels, nclasses,
                    block_args = (; cardinality, base_width))
    if pretrain
        loadpretrain!(layers, string("resnext", depth, "_", cardinality, "x", base_width))
    end
    return Res2NeXt(layers)
end

backbone(m::Res2NeXt) = m.layers[1]
classifier(m::Res2NeXt) = m.layers[2]
