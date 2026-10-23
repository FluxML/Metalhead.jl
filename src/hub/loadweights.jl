# Map a Flux model onto PyTorch `state_dict` keys by walking its structure.
# Ported models mirror the PyTorch module tree, so key names follow from the walk:
# struct fields and NamedTuple entries give their names, Tuple/Vector entries their
# 0-based index, and Flux containers (Chain, SkipConnection, Parallel) add no segment.

"""
    TorchParam(key, array, kind)

A parameter or buffer of a Flux model, with its PyTorch `state_dict` key.
`kind` selects the transform from the checkpoint's Julia-layout tensor:
`:plain` (none), `:dense` (transpose) or `:conv` (flip the spatial dimensions).
"""
struct TorchParam
    key::String
    array::AbstractArray
    kind::Symbol
end

join_key(prefix, name) = isempty(prefix) ? string(name) : string(prefix, ".", name)

hasarray(b) = b isa AbstractArray

# Parameters of Flux leaf layers, as (torch name, array, kind).
function layer_params(l::Dense)
    return [("weight", l.weight, :dense);
            hasarray(l.bias) ? [("bias", l.bias, :plain)] : []]
end
function layer_params(l::Union{Conv, ConvTranspose})
    return [("weight", l.weight, :conv); hasarray(l.bias) ? [("bias", l.bias, :plain)] : []]
end
function layer_params(l::BatchNorm)
    ps = l.affine ? [("weight", l.γ, :plain), ("bias", l.β, :plain)] : []
    l.track_stats &&
        append!(ps, [("running_mean", l.μ, :plain), ("running_var", l.σ², :plain)])
    return ps
end
function layer_params(l::LayerNorm)
    return l.affine ? [("weight", l.diag.scale, :plain), ("bias", l.diag.bias, :plain)] : []
end
function layer_params(l::GroupNorm)
    return l.affine ? [("weight", l.γ, :plain), ("bias", l.β, :plain)] : []
end
layer_params(l::Embedding) = [("weight", l.weight, :plain)]
layer_params(_) = nothing

"""
    torch_params(model; prefix = "") -> (params::Vector{TorchParam}, aliases::Vector{Pair{String,String}})

Enumerate the parameters and buffers of `model` with their PyTorch `state_dict` keys.
An array reached twice (tied weights) is listed once; later keys are returned as `aliases`.
"""
function torch_params(model; prefix::AbstractString = "")
    params = TorchParam[]
    aliases = Pair{String, String}[]
    seen = IdDict{Any, String}()
    function record!(key, array, kind)
        if haskey(seen, array)
            push!(aliases, key => seen[array])
        else
            seen[array] = key
            push!(params, TorchParam(key, array, kind))
        end
    end
    function walk(x, key)
        ps = layer_params(x)
        if ps !== nothing
            for (name, array, kind) in ps
                record!(join_key(key, name), array, kind)
            end
        elseif x isa AbstractArray{<:Number}
            record!(key, x, :plain)
        elseif x isa Chain
            walk(x.layers, key)
        elseif x isa SkipConnection
            walk(x.layers, key)
            walk(x.connection, key)
        elseif x isa Parallel
            walk(x.connection, key)
            walk(x.layers, key)
        elseif x isa Union{Tuple, AbstractVector}
            for (i, c) in enumerate(x)
                walk(c, join_key(key, i - 1))
            end
        elseif x isa NamedTuple
            for (k, c) in pairs(x)
                walk(c, join_key(key, k))
            end
        elseif x isa AbstractDict
            error("dictionaries are not supported in ported models (at `$key`)")
        elseif !Functors.isleaf(x)
            if parentmodule(typeof(x)) === Flux && !isempty(Flux.trainables(x))
                error("Flux layer $(nameof(typeof(x))) at `$key` has no PyTorch mapping; use a layer with one or port it explicitly")
            end
            for (k, c) in pairs(Functors.children(x))
                walk(c, join_key(key, k))
            end
        end
    end
    walk(model, prefix)
    return params, aliases
end

transform(::Val{:plain}, x) = x
transform(::Val{:dense}, x) = permutedims(x, (2, 1))
transform(::Val{:conv}, x) = reverse(x; dims = Tuple(1:(ndims(x) - 2)))

const DEFAULT_IGNORE = [r"num_batches_tracked$"]

function levenshtein(a::AbstractString, b::AbstractString)
    a, b = collect(a), collect(b)
    d = collect(0:length(b))
    for i in eachindex(a)
        prev, d[1] = d[1], i
        for j in eachindex(b)
            prev, d[j + 1] = d[j + 1], min(d[j + 1] + 1, d[j] + 1, prev + (a[i] != b[j]))
        end
    end
    return d[end]
end

function closest(key, candidates)
    return isempty(candidates) ? nothing : argmin(c -> levenshtein(key, c), candidates)
end

"""
    load_state!(model, ckpt::Checkpoint; prefix = "", ignore = DEFAULT_IGNORE, skip = nothing, strict = true)

Copy the tensors of `ckpt` into `model`, matching PyTorch `state_dict` keys derived from the
model structure (see [`torch_params`](@ref)). Keys matching a regex in `ignore` may be absent
from the model; parameters whose key starts with `skip` keep their current values.
With `strict = true`, missing keys, unused checkpoint keys and shape mismatches throw.
"""
function load_state!(
        model, ckpt::Checkpoint; prefix::AbstractString = "", ignore = DEFAULT_IGNORE,
        skip::Union{Nothing, AbstractString} = nothing, strict::Bool = true)
    params, aliases = torch_params(model; prefix)
    skipped(k) = skip !== nothing && (k == skip || startswith(k, skip * "."))
    used = Set{String}()
    missing_keys = String[]
    mismatched = String[]
    for p in params
        skipped(p.key) && continue
        if !haskey(ckpt, p.key)
            push!(missing_keys, p.key)
            continue
        end
        x = transform(Val(p.kind), ckpt[p.key])
        push!(used, p.key)
        if size(x) != size(p.array)
            push!(mismatched,
                "$(p.key): checkpoint has PyTorch shape $(torch_shape(ckpt, p.key)), model expects Julia shape $(size(p.array))")
            continue
        end
        copyto!(p.array, x)
    end
    for (key, _) in aliases
        haskey(ckpt, key) && push!(used, key)
    end
    unexpected = sort!([k
                        for k in keys(ckpt)
                        if k ∉ used && !skipped(k) && !any(r -> occursin(r, k), ignore)])
    if strict && !(isempty(missing_keys) && isempty(unexpected) && isempty(mismatched))
        lines = String[]
        for k in missing_keys
            c = closest(k, unexpected)
            push!(lines, "  missing:    $k" *
                         (c === nothing ? "" : "  (closest unused checkpoint key: $c)"))
        end
        foreach(k -> push!(lines, "  unused:     $k"), unexpected)
        foreach(m -> push!(lines, "  mismatched: $m"), mismatched)
        error("checkpoint does not match the model:\n" * join(lines, "\n"))
    end
    return (; loaded = length(used) - length(mismatched),
        missing = missing_keys, unexpected, mismatched)
end
