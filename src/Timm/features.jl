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
    return output_fmt in allowed ||
           error("`output_fmt = $(repr(output_fmt))` must be one of $(join(repr.(allowed), ", "))")
end
ChainRulesCore.@non_differentiable check_output_fmt(::Any...)

# The outputs of `stages` applied in turn to `x`, stopping after `stop` stages.
function stage_outputs(stages, x, stop = length(stages))
    layers = stages isa NamedTuple ? values(stages) : stages
    return stop == 0 ? () : Flux.activations(Chain(layers[1:stop]...), x)
end

function stage_outputs(stages::AbstractVector, x, stop = length(stages))
    stop == 0 && return ()
    # Include x in the sequence: accumulate's rule does not differentiate `init`.
    return accumulate((y, layer) -> layer(y), vcat([x], stages[1:stop]))[2:end]
end

function sequential_intermediates(stages, x, ends; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, output_fmt = "NCHW", intermediates_only::Bool = false)
    check_output_fmt(output_fmt)
    take, max_index = feature_take_indices(length(ends), indices)
    outs = stage_outputs(stages, x, stop_early ? ends[max_index] : length(stages))
    intermediates = [outs[ends[i]] for i in take]
    return intermediates_only ? intermediates : (last(outs), intermediates)
end

# (C, L, B) tokens of a (W, H) grid as a WHCN feature map, timm's NLC -> NCHW reshape.
function tokens_to_nchw(y, (w, h))
    return permutedims(reshape(y, size(y, 1), w, h, size(y, 3)), (2, 3, 1, 4))
end
