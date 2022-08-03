"""
    efficientnet(scalings, block_configs; max_width::Integer = 1280,
                 inchannels::Integer = 3, nclasses::Integer = 1000)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).

# Arguments

  - `scalings`: global width and depth scaling (given as a tuple)

  - `block_configs`: configuration for each inverted residual block,
    given as a vector of tuples with elements:
    
      + `n`: number of block repetitions (will be scaled by global depth scaling)
      + `k`: kernel size
      + `s`: kernel stride
      + `e`: expansion ratio
      + `i`: block input channels (will be scaled by global width scaling)
      + `o`: block output channels (will be scaled by global width scaling)
  - `inchannels`: number of input channels
  - `nclasses`: number of output classes
  - `max_width`: maximum number of output channels before the fully connected
    classification blocks
"""
function efficientnet(scalings::NTuple{2, Real},
                      block_configs::AbstractVector{NTuple{6, Int}};
                      max_width::Integer = 1280, inchannels::Integer = 3,
                      nclasses::Integer = 1000)
    wscale, dscale = scalings
    scalew(w) = wscale ≈ 1 ? w : ceil(Int64, wscale * w)
    scaled(d) = dscale ≈ 1 ? d : ceil(Int64, dscale * d)
    out_channels = _round_channels(scalew(32), 8)
    stem = conv_norm((3, 3), inchannels, out_channels, swish; bias = false, stride = 2,
                     pad = SamePad())
    blocks = []
    for (n, k, s, e, i, o) in block_configs
        in_channels = _round_channels(scalew(i), 8)
        out_channels = _round_channels(scalew(o), 8)
        repeats = scaled(n)
        push!(blocks,
              invertedresidual((k, k), in_channels, out_channels, swish; expansion = e,
                               stride = s, reduction = 4))
        for _ in 1:(repeats - 1)
            push!(blocks,
                  invertedresidual((k, k), out_channels, out_channels, swish; expansion = e,
                                   stride = 1, reduction = 4))
        end
    end
    head_out_channels = _round_channels(max_width, 8)
    append!(blocks,
            conv_norm((1, 1), out_channels, head_out_channels, swish;
                      bias = false, pad = SamePad()))
    return Chain(Chain(stem..., blocks...), create_classifier(head_out_channels, nclasses))
end

# n: # of block repetitions
# k: kernel size k x k
# s: stride
# e: expantion ratio
# i: block input channels
# o: block output channels
const EFFICIENTNET_BLOCK_CONFIGS = [
    # (n, k, s, e, i, o)
    (1, 3, 1, 1, 32, 16),
    (2, 3, 2, 6, 16, 24),
    (2, 5, 2, 6, 24, 40),
    (3, 3, 2, 6, 40, 80),
    (3, 5, 1, 6, 80, 112),
    (4, 5, 2, 6, 112, 192),
    (1, 3, 1, 6, 192, 320),
]

# w: width scaling
# d: depth scaling
# r: image resolution
# Data is organised as (r, (w, d))
const EFFICIENTNET_GLOBAL_CONFIGS = Dict(:b0 => (224, (1.0, 1.0)),
                                         :b1 => (240, (1.0, 1.1)),
                                         :b2 => (260, (1.1, 1.2)),
                                         :b3 => (300, (1.2, 1.4)),
                                         :b4 => (380, (1.4, 1.8)),
                                         :b5 => (456, (1.6, 2.2)),
                                         :b6 => (528, (1.8, 2.6)),
                                         :b7 => (600, (2.0, 3.1)),
                                         :b8 => (672, (2.2, 3.6)))

"""
    EfficientNet(config::Symbol; pretrain::Bool = false)

Create an EfficientNet model ([reference](https://arxiv.org/abs/1905.11946v5)).
See also [`efficientnet`](#).

# Arguments

  - `config`: name of default configuration
    (can be `:b0`, `:b1`, `:b2`, `:b3`, `:b4`, `:b5`, `:b6`, `:b7`, `:b8`)
  - `pretrain`: set to `true` to load the pre-trained weights for ImageNet
"""
struct EfficientNet
    layers::Any
end
@functor EfficientNet

function EfficientNet(config::Symbol; pretrain::Bool = false)
    _checkconfig(config, keys(EFFICIENTNET_GLOBAL_CONFIGS))
    model = efficientnet(EFFICIENTNET_GLOBAL_CONFIGS[config][2], EFFICIENTNET_BLOCK_CONFIGS)
    if pretrain
        loadpretrain!(model, string("efficientnet-", config))
    end
    return model
end

(m::EfficientNet)(x) = m.layers(x)

backbone(m::EfficientNet) = m.layers[1]
classifier(m::EfficientNet) = m.layers[2]
