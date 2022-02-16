"""
    ConvNeXtBlock(planes, drop_path = 0.0f32, λ = 1f-6)

Creates a single block of ConvNeXt.
([reference](https://arxiv.org/pdf/2201.03545))

# Arguments:
- `planes`: number of input channels.
- `drop_path`: Stochastic depth rate.
- `λ`: Init value for [LayerScale](https://arxiv.org/pdf/2103.17239)
"""
function ConvNeXtBlock(planes, drop_path = 0.0f32, λ = 1f-6)
  γ = λ > 0 ? Flux.ones32(planes) * λ : nothing
  droppath = drop_path > 0 ? DropPath(drop_path) : identity
  layers = SkipConnection(Chain(DepthwiseConv((7, 7), planes => planes, pad = 3), 
                          x -> permutedims(x, (3, 1, 2, 4)),
                          LayerNorm(planes; ϵ = 1f-6),
                          mlpblock(planes, 4 * planes),
                          x -> isnothing(γ) ? x : x .* γ,
                          x -> permutedims(x, (2, 3, 1, 4)),
                          droppath), +)
  return layers
end

"""
    convnext(; inchannels = 3, depths = [3, 3, 9, 3], planes = [96, 192, 384, 768], 
               drop_path_rate = 0.0f32, λ = 1f-6, nclasses = 1000)

Creates the layers for a ConvNeXt model.
([reference](https://arxiv.org/pdf/2201.03545))

# Arguments:
- `inchannels`: number of input channels.
- `depths`: list with configuration for depth of each block
- `planes`: list with configuration for number of output channels in each block
- `drop_path_rate`: Stochastic depth rate.
- `λ`: Init value for [LayerScale](https://arxiv.org/pdf/2103.17239)
- `nclasses`: number of output classes
"""
function convnext(; inchannels = 3, depths = [3, 3, 9, 3], planes = [96, 192, 384, 768], 
                    drop_path_rate = 0.0f32, λ = 1fe-6, nclasses = 1000)
  
  layers = []
  stem = Chain(Conv((4, 4), inchannels => planes[1]; stride = 4),
               ChannelLayerNorm(planes[1]; ϵ = 1f-6))
  push!(layers, stem)
  dp_rates = LinRange{Float32}(0.0f32, drop_path_rate, sum(depths))
  cur = 0
  for i in 2:8
    m = i ÷ 2
    if i % 2 == 1
      downsample_layer = Chain(ChannelLayerNorm(planes[m]; ϵ = 1f-6),
                               Conv((2, 2), planes[m] => planes[m + 1]; stride = 2))
      push!(layers, downsample_layer)
    else
      append!(layers, [ConvNeXtBlock(planes[m], dp_rates[cur + j], λ) for j in 1:depths[m]])
      cur += depths[m]
    end
  end

  head = Chain(GlobalMeanPool(),
               x -> dropdims(x; dims = (1, 2)),
               LayerNorm(planes[end]),
               Dense(planes[end], nclasses))
  return Chain(layers..., head)
end

struct ConvNeXt
  layers
end

"""
    ConvNeXt(; inchannels = 3, drop_path_rate = 0.0f32, λ = 1f-6, nclasses = 1000)

Creates a ConvNeXt model.
([reference](https://arxiv.org/pdf/2201.03545))

# Arguments:
- `inchannels`: number of input channels.
- `drop_path_rate`: Stochastic depth rate.
- `λ`: Init value for [LayerScale](https://arxiv.org/pdf/2103.17239)
- `nclasses`: number of output classes
"""
function ConvNeXt(; inchannels = 3, drop_path_rate = 0.0f32, λ = 1f-6, nclasses = 1000)
  layers = convnext(; inchannels, drop_path_rate, λ, nclasses)
  return ConvNeXt(layers)
end

(m::ConvNeXt)(x) = m.layers(x)

@functor ConvNeXt
