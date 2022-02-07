struct MBConv{E, D, X, P}
  expansion::E
  depthwise::D
  excitation::X
  projection::P

  do_expansion::Bool
  do_excitation::Bool
  do_skip::Bool
end
Flux.@functor MBConv

"""
    MBConv(
      in_channels, out_channels, kernel, stride;
      expansion_ratio, se_ratio)

Mobile Inverted Residual Bottleneck Block
([reference](https://arxiv.org/abs/1801.04381)).

# Arguments
- `in_channels`: Number of input channels.
- `out_channels`: Number of output channels.
- `expansion_ratio`:
  Expansion ratio defines the number of output channels.
  Set to `1` to disable expansion phase.
  `out_channels = input_channels * expansion_ratio`.
- `kernel`: Size of the kernel for the depthwise conv phase.
- `stride`: Size of the stride for the depthwise conv phase.
- `se_ratio`:
  Squeeze-Excitation ratio. Should be in `(0, 1]` range.
  Set to `-1` to disable.
"""
function MBConv(
  in_channels, out_channels, kernel, stride;
  expansion_ratio, se_ratio = 0.25,
)
  do_skip = stride == 1 && in_channels == out_channels
  do_expansion, do_excitation = expansion_ratio != 1, 0 < se_ratio ≤ 1
  pad, bias = SamePad(), false

  mid_channels = ceil(Int, in_channels * expansion_ratio)
  expansion = do_expansion ?
    Chain(
      Conv((1, 1), in_channels=>mid_channels; bias, pad),
      BatchNorm(mid_channels, swish)) :
    identity

  depthwise = Chain(
    Conv(kernel, mid_channels=>mid_channels; bias, stride, pad, groups=mid_channels),
    BatchNorm(mid_channels, swish))

  if do_excitation
    n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
    excitation = Chain(
      AdaptiveMeanPool((1, 1)),
      Conv((1, 1), mid_channels=>n_squeezed_channels, swish; pad),
      Conv((1, 1), n_squeezed_channels=>mid_channels; pad))
  else
    excitation = identity
  end

  projection = Chain(
    Conv((1, 1), mid_channels=>out_channels; pad, bias),
    BatchNorm(out_channels))
  MBConv(
    expansion, depthwise, excitation, projection, do_expansion,
    do_excitation, do_skip)
end

function (m::MBConv)(x)
  o = m.depthwise(m.expansion(x))

  if m.do_excitation
    o = σ.(m.excitation(o)) .* o
  end
  o = m.projection(o)
  if m.do_skip
    o = o + x
  end
  o
end
