function efficientnet(scalings, block_config;
                      inchannels = 3, nclasses = 1000, max_width = 1280)
    wscale, dscale = scalings
    out_channels = _round_channels(32, 8)
    stem = Chain(Conv((3, 3), inchannels => out_channels;
                      bias = false, stride = 2, pad = SamePad()),
                 BatchNorm(out_channels, swish))

    blocks = []
    for (n, k, s, e, i, o) in block_config
        in_channels = round_filter(i, 8)
        out_channels = round_filter(o, 8)
        repeat = dscale ≈ 1 ? n : ceil(Int64, dscale * n)

        push!(blocks,
              invertedresidual(k, in_channels, in_channels * e, out_channels, swish;
                               stride = s, reduction = 4))
        for _ in 1:(repeat - 1)
            push!(blocks,
                  invertedresidual(k, out_channels, out_channels * e, out_channels, swish;
                                   stride = 1, reduction = 4))
        end
    end
    blocks = Chain(blocks...)

    head_out_channels = _round_channels(max_width, 8)
    head = Chain(Conv((1, 1), out_channels => head_out_channels;
                      bias = false, pad = SamePad()),
                BatchNorm(head_out_channels, swish))

    top = Dense(head_out_channels, nclasses)

    return Chain(Chain(stem, blocks, head),
                Chain(AdaptiveMeanPool((1, 1)), MLUtils.flatten, top))
end

# n: # of block repetitions
# k: kernel size k x k
# s: stride
# e: expantion ratio
# i: block input channels
# o: block output channels
const efficientnet_block_configs = [
#   (n, k, s, e,   i,   o)
    (1, 3, 1, 1,  32,  16),
    (2, 3, 2, 6,  16,  24),
    (2, 5, 2, 6,  24,  40),
    (3, 3, 2, 6,  40,  80),
    (3, 5, 1, 6,  80, 112),
    (4, 5, 2, 6, 112, 192),
    (1, 3, 1, 6, 192, 320)
]

# w: width scaling
# d: depth scaling
# r: image resolution
const efficientnet_global_configs = Dict(
#          (  r, (  w,   d))
    :b0 => (224, (1.0, 1.0)),
    :b1 => (240, (1.0, 1.1)),
    :b2 => (260, (1.1, 1.2)),
    :b3 => (300, (1.2, 1.4)),
    :b4 => (380, (1.4, 1.8)),
    :b5 => (456, (1.6, 2.2)),
    :b6 => (528, (1.8, 2.6)),
    :b7 => (600, (2.0, 3.1)),
    :b8 => (672, (2.2, 3.6))
)

struct EfficientNet
  layers::Any
end

function EfficientNet(scalings, block_config;
                      inchannels = 3, nclasses = 1000, max_width = 1280)
  layers = efficientnet(scalings, block_config;
                        inchannels = inchannels,
                        nclasses = nclasses,
                        max_width = max_width)
  return EfficientNet(layers)
end

@functor EfficientNet

(m::EfficientNet)(x) = m.layers(x)

backbone(m::EfficientNet) = m.layers[1]
classifier(m::EfficientNet) = m.layers[2]

function EfficientNet(name::Symbol; pretrain = false)
    @assert name in keys(efficientnet_global_configs)
        "`name` must be one of $(sort(collect(keys(efficientnet_global_configs))))"

    model = EfficientNet(efficientnet_global_configs[name]..., efficientnet_block_configs)
    pretrain && loadpretrain!(model, string("efficientnet-", name))

    return model
end
