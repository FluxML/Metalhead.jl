# Ported from timm.models._features (timm 1.0.30, Apache-2.0, Ross Wightman).

# The sorted positions in 1:n selected by `indices` of forward_intermediates, and the last.
function feature_take_indices(n::Integer, indices)
    take = indices === nothing ? collect(1:n) :
           indices isa Integer ? collect((n - indices + 1):n) :
           sort!(unique!(collect(Int, indices)))
    if isempty(take) || !all(in(1:n), take)
        error("`indices = $(repr(indices))` must select from the $n intermediates (1-based)")
    end
    return take, last(take)
end
ChainRulesCore.@non_differentiable feature_take_indices(::Any...)

function check_output_fmt(output_fmt, allowed = ("NCHW",))
    output_fmt in allowed ||
        error("`output_fmt = $(repr(output_fmt))` must be one of $(join(repr.(allowed), ", "))")
end
ChainRulesCore.@non_differentiable check_output_fmt(::Any...)

# The outputs of `stages` applied in turn to `x`, as a tuple, stopping after `stop` stages.
function stage_outputs(stages, x, stop = length(stages))
    return stop == 0 ? () : Flux.activations(Chain(stages[1:stop]...), x)
end

# (C, L, B) tokens of a (W, H) grid as a WHCN feature map, timm's NLC -> NCHW reshape.
function tokens_to_nchw(y, (w, h))
    return permutedims(reshape(y, size(y, 1), w, h, size(y, 3)), (2, 3, 1, 4))
end
