"""
    convnextblock(planes, drop_path_rate = 0., λ = 1f-6)

Creates a single block of ConvNeXt.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `planes`: number of input channels.
  - `drop_path_rate`: Stochastic depth rate.
  - `λ`: Initial value for [`LayerScale`](#)
"""
function convnextblock(planes, drop_path_rate = 0.0, λ = 1.0f-6)
    layers = SkipConnection(Chain(DepthwiseConv((7, 7), planes => planes; pad = 3),
                                  swapdims((3, 1, 2, 4)),
                                  LayerNorm(planes; ϵ = 1.0f-6),
                                  mlp_block(planes, 4 * planes),
                                  LayerScale(planes, λ),
                                  swapdims((2, 3, 1, 4)),
                                  DropPath(drop_path_rate)), +)
    return layers
end

"""
    convnext(depths, planes; inchannels = 3, drop_path_rate = 0., λ = 1f-6, nclasses = 1000)

Creates the layers for a ConvNeXt model.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `inchannels`: number of input channels.
  - `depths`: list with configuration for depth of each block
  - `planes`: list with configuration for number of output channels in each block
  - `drop_path_rate`: Stochastic depth rate.
  - `λ`: Initial value for [`LayerScale`](#)
    ([reference](https://arxiv.org/abs/2103.17239))
  - `nclasses`: number of output classes
"""
function convnext(depths, planes; inchannels = 3, drop_path_rate = 0.0, λ = 1.0f-6,
                  nclasses = 1000)
    @assert length(depths) == length(planes)
    "`planes` should have exactly one value for each block"
    downsample_layers = []
    stem = Chain(Conv((4, 4), inchannels => planes[1]; stride = 4),
                 ChannelLayerNorm(planes[1]))
    push!(downsample_layers, stem)
    for m in 1:(length(depths) - 1)
        downsample_layer = Chain(ChannelLayerNorm(planes[m]),
                                 Conv((2, 2), planes[m] => planes[m + 1]; stride = 2))
        push!(downsample_layers, downsample_layer)
    end
    stages = []
    dp_rates = linear_scheduler(drop_path_rate; depth = sum(depths))
    cur = 0
    for i in eachindex(depths)
        push!(stages, [convnextblock(planes[i], dp_rates[cur + j], λ) for j in 1:depths[i]])
        cur += depths[i]
    end
    backbone = collect(Iterators.flatten(Iterators.flatten(zip(downsample_layers, stages))))
    head = Chain(GlobalMeanPool(),
                 MLUtils.flatten,
                 LayerNorm(planes[end]),
                 Dense(planes[end], nclasses))
    return Chain(Chain(backbone), head)
end

# Configurations for ConvNeXt models
convnext_configs = Dict(:tiny => ([3, 3, 9, 3], [96, 192, 384, 768]),
                        :small => ([3, 3, 27, 3], [96, 192, 384, 768]),
                        :base => ([3, 3, 27, 3], [128, 256, 512, 1024]),
                        :large => ([3, 3, 27, 3], [192, 384, 768, 1536]),
                        :xlarge => ([3, 3, 27, 3], [256, 512, 1024, 2048]))

struct ConvNeXt
    layers::Any
end
@functor ConvNeXt

"""
    ConvNeXt(mode::Symbol = :base; inchannels = 3, drop_path_rate = 0., λ = 1f-6, nclasses = 1000)

Creates a ConvNeXt model.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `inchannels`: The number of channels in the input.
  - `drop_path_rate`: Stochastic depth rate.
  - `λ`: Init value for [LayerScale](https://arxiv.org/abs/2103.17239)
  - `nclasses`: number of output classes

See also [`Metalhead.convnext`](#).
"""
function ConvNeXt(mode::Symbol = :base; inchannels = 3, drop_path_rate = 0.0, λ = 1.0f-6,
                  nclasses = 1000)
    _checkconfig(mode, keys(convnext_configs))
    layers = convnext(convnext_configs[mode]...; inchannels, drop_path_rate, λ, nclasses)
    return ConvNeXt(layers)
end

(m::ConvNeXt)(x) = m.layers(x)

backbone(m::ConvNeXt) = m.layers[1]
classifier(m::ConvNeXt) = m.layers[2]
