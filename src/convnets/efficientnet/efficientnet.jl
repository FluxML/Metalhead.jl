include("params.jl")
include("mb.jl")

struct EfficientNet{S, B, H, P, F}
  stem::S
  blocks::B

  head::H
  pooling::P
  top::F
end
Flux.@functor EfficientNet

"""
    EfficientNet(block_params, global_params; in_channels = 3)

Construct an EfficientNet model
([reference](https://arxiv.org/abs/1905.11946)).
"""
function EfficientNet(
  model_name, block_params, global_params; in_channels, n_classes, pretrain,
)
  pad, bias = SamePad(), false
  out_channels = round_filter(32, global_params)
  stem = Chain(
    Conv((3, 3), in_channels=>out_channels; bias, stride=2, pad),
    BatchNorm(out_channels, swish))

  blocks = MBConv[]
  for bp in block_params
    in_channels = round_filter(bp.in_channels, global_params)
    out_channels = round_filter(bp.out_channels, global_params)
    repeat = global_params.depth_coef ≈ 1 ?
      bp.repeat : ceil(Int64, global_params.depth_coef * bp.repeat)

    push!(blocks, MBConv(
      in_channels, out_channels, bp.kernel, bp.stride;
      expansion_ratio=bp.expansion_ratio))
    for _ in 1:(repeat - 1)
      push!(blocks, MBConv(
        out_channels, out_channels, bp.kernel, 1;
        expansion_ratio=bp.expansion_ratio))
    end
  end
  blocks = Chain(blocks...)

  head_out_channels = round_filter(1280, global_params)
  head = Chain(
    Conv((1, 1), out_channels=>head_out_channels; bias, pad),
    BatchNorm(head_out_channels, swish))

  top = n_classes ≡ nothing ?
    identity : (Dense(head_out_channels, n_classes) ∘ Flux.flatten)
  model = EfficientNet(stem, blocks, head, AdaptiveMeanPool((1, 1)), top)
  pretrain && loadpretrain!(model, "EfficientNet" * model_name)
  model
end

EfficientNet(model_name::String; in_channels = 3, n_classes = 1000, pretrain = false) =
  EfficientNet(model_name, get_efficientnet_params(model_name)...; in_channels, n_classes, pretrain)

(m::EfficientNet)(x) = m.top(m.pooling(m.head(m.blocks(m.stem(x)))))
