"""
    efficientnet(scalings, block_config;
                 inchannels = 3, nclasses = 1000, max_width = 1280)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).

# Arguments

- `scalings`: global width and depth scaling (given as a tuple)
- `block_config`: configuration for each inverted residual block,
                  given as a vector of tuples with elements:
    - `n`: number of block repetitions (will be scaled by global depth scaling)
    - `k`: kernel size
    - `s`: kernel stride
    - `e`: expansion ratio
    - `i`: block input channels
    - `o`: block output channels (will be scaled by global width scaling)
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `max_width`: maximum number of output channels before the fully connected
               classification blocks
"""
function efficientnet(scalings, block_config;
                      inchannels = 3, nclasses = 1000, max_width = 1280)
    wscale, dscale = scalings
    scalew(w) = wscale ≈ 1 ? w : ceil(Int64, wscale * w)
    scaled(d) = dscale ≈ 1 ? d : ceil(Int64, dscale * d)

    out_channels = _round_channels(scalew(32), 8)
    stem = conv_bn((3, 3), inchannels, out_channels, swish;
                   bias = false, stride = 2, pad = SamePad())

    blocks = []
    for (n, k, s, e, i, o) in block_config
        in_channels = _round_channels(scalew(i), 8)
        out_channels = _round_channels(scalew(o), 8)
        repeats = scaled(n)

        push!(blocks,
              invertedresidual(k, in_channels, in_channels * e, out_channels, swish;
                               stride = s, reduction = 4))
        for _ in 1:(repeats - 1)
            push!(blocks,
                  invertedresidual(k, out_channels, out_channels * e, out_channels, swish;
                                   stride = 1, reduction = 4))
        end
    end
    blocks = Chain(blocks...)

    head_out_channels = _round_channels(max_width, 8)
    head = conv_bn((1, 1), out_channels, head_out_channels, swish;
                   bias = false, pad = SamePad())

    top = Dense(head_out_channels, nclasses)

    return Chain(Chain(stem..., blocks, head...),
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

"""
    EfficientNet(scalings, block_config;
                 inchannels = 3, nclasses = 1000, max_width = 1280)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).
See also [`efficientnet`](#).

# Arguments

- `scalings`: global width and depth scaling (given as a tuple)
- `block_config`: configuration for each inverted residual block,
                  given as a vector of tuples with elements:
    - `n`: number of block repetitions (will be scaled by global depth scaling)
    - `k`: kernel size
    - `s`: kernel stride
    - `e`: expansion ratio
    - `i`: block input channels
    - `o`: block output channels (will be scaled by global width scaling)
- `inchannels`: number of input channels
- `nclasses`: number of output classes
- `max_width`: maximum number of output channels before the fully connected
               classification blocks
"""
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

"""
    EfficientNet(name::Symbol; pretrain = false)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).
See also [`efficientnet`](#).

# Arguments

- `name`: name of default configuration
          (can be `:b0`, `:b1`, `:b2`, `:b3`, `:b4`, `:b5`, `:b6`, `:b7`, `:b8`)
- `pretrain`: set to `true` to load the pre-trained weights for ImageNet
"""
function EfficientNet(name::Symbol; pretrain = false)
    @assert name in keys(efficientnet_global_configs)
        "`name` must be one of $(sort(collect(keys(efficientnet_global_configs))))"

    model = EfficientNet(efficientnet_global_configs[name][2], efficientnet_block_configs)
    pretrain && loadpretrain!(model, string("efficientnet-", name))

    return model
end
