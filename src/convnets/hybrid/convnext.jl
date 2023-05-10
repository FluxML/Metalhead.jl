"""
    convnextblock(planes::Integer, stochastic_depth_prob = 0.0, layerscale_init = 1.0f-6)

Creates a single block of ConvNeXt.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `planes`: number of input channels.
  - `stochastic_depth_prob`: Stochastic depth probability.
  - `layerscale_init`: Initial value for [`Metalhead.LayerScale`](@ref)
"""
function convnextblock(planes::Integer, stochastic_depth_prob = 0.0,
                       layerscale_init = 1.0f-6)
    return SkipConnection(Chain(DepthwiseConv((7, 7), planes => planes; pad = 3),
                                swapdims((3, 1, 2, 4)),
                                LayerNorm(planes; ϵ = 1.0f-6),
                                mlp_block(planes, 4 * planes),
                                LayerScale(planes, layerscale_init),
                                swapdims((2, 3, 1, 4)),
                                StochasticDepth(stochastic_depth_prob)), +)
end

"""
    build_convnext(depths::AbstractVector{<:Integer}, planes::AbstractVector{<:Integer};
                   stochastic_depth_prob = 0.0, layerscale_init = 1.0f-6,
                   inchannels::Integer = 3, nclasses::Integer = 1000)

Creates the layers for a ConvNeXt model.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `depths`: list with configuration for depth of each block
  - `planes`: list with configuration for number of output channels in each block
  - `stochastic_depth_prob`: Stochastic depth probability.
  - `layerscale_init`: Initial value for [`LayerScale`](@ref)
    ([reference](https://arxiv.org/abs/2103.17239))
  - `inchannels`: number of input channels.
  - `nclasses`: number of output classes
"""
function build_convnext(depths::AbstractVector{<:Integer},
                        planes::AbstractVector{<:Integer};
                        stochastic_depth_prob = 0.0, layerscale_init = 1.0f-6,
                        inchannels::Integer = 3, nclasses::Integer = 1000)
    @assert length(depths) == length(planes)
    "`planes` should have exactly one value for each block"
    downsample_layers = []
    push!(downsample_layers,
          Chain(conv_norm((4, 4), inchannels, planes[1]; stride = 4,
                          norm_layer = ChannelLayerNorm)...))
    for m in 1:(length(depths) - 1)
        push!(downsample_layers,
              Chain(conv_norm((2, 2), planes[m], planes[m + 1]; stride = 2,
                              norm_layer = ChannelLayerNorm, revnorm = true)...))
    end
    stages = []
    sdschedule = linear_scheduler(stochastic_depth_prob; depth = sum(depths))
    cur = 0
    for i in eachindex(depths)
        push!(stages,
              [convnextblock(planes[i], sdschedule[cur + j], layerscale_init)
               for j in 1:depths[i]])
        cur += depths[i]
    end
    backbone = collect(Iterators.flatten(Iterators.flatten(zip(downsample_layers, stages))))
    classifier = Chain(GlobalMeanPool(), MLUtils.flatten,
                       LayerNorm(planes[end]), Dense(planes[end], nclasses))
    return Chain(Chain(backbone...), classifier)
end

"""
    convnext(config::Symbol; stochastic_depth_prob = 0.0, layerscale_init = 1.0f-6,
             inchannels::Integer = 3, nclasses::Integer = 1000)

Creates a ConvNeXt model.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `config`: The size of the model, one of `tiny`, `small`, `base`, `large` or `xlarge`.
  - `stochastic_depth_prob`: Stochastic depth probability.
  - `layerscale_init`: Initial value for [`LayerScale`](@ref)
    ([reference](https://arxiv.org/abs/2103.17239))
  - `inchannels`: number of input channels.
  - `nclasses`: number of output classes
"""
function convnext(config::Symbol; stochastic_depth_prob = 0.0, layerscale_init = 1.0f-6,
                  inchannels::Integer = 3, nclasses::Integer = 1000)
    return build_convnext(CONVNEXT_CONFIGS[config]...; stochastic_depth_prob,
                          layerscale_init, inchannels, nclasses)
end

# Configurations for ConvNeXt models
const CONVNEXT_CONFIGS = Dict(:tiny => ([3, 3, 9, 3], [96, 192, 384, 768]),
                              :small => ([3, 3, 27, 3], [96, 192, 384, 768]),
                              :base => ([3, 3, 27, 3], [128, 256, 512, 1024]),
                              :large => ([3, 3, 27, 3], [192, 384, 768, 1536]),
                              :xlarge => ([3, 3, 27, 3], [256, 512, 1024, 2048]))

"""
    ConvNeXt(config::Symbol; pretrain::Bool = true, inchannels::Integer = 3,
             nclasses::Integer = 1000)

Creates a ConvNeXt model.
([reference](https://arxiv.org/abs/2201.03545))

# Arguments

  - `config`: The size of the model, one of `tiny`, `small`, `base`, `large` or `xlarge`.
  - `pretrain`: set to `true` to load pre-trained weights for ImageNet
  - `inchannels`: number of input channels
  - `nclasses`: number of output classes

!!! warning
    
    `ConvNeXt` does not currently support pretrained weights.

See also [`Metalhead.convnext`](@ref).
"""
struct ConvNeXt
    layers::Any
end
@functor ConvNeXt

function ConvNeXt(config::Symbol; pretrain::Bool = false, inchannels::Integer = 3,
                  nclasses::Integer = 1000)
    _checkconfig(config, keys(CONVNEXT_CONFIGS))
    layers = convnext(config; inchannels, nclasses)
    model = ConvNeXt(layers)
    if pretrain
        loadpretrain!(model, "convnext_$config")
    end
    return model
end

(m::ConvNeXt)(x) = m.layers(x)

backbone(m::ConvNeXt) = m.layers[1]
classifier(m::ConvNeXt) = m.layers[2]
