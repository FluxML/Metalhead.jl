# Numerical parity of Metalhead ports against PyTorch, on the fixtures written by porting/dump.py
# and porting/dump_reference.py. Used by the test suite and by porting/check.jl.
module Parity

using Metalhead, Flux, Functors, LinearAlgebra, SafeTensors
using Metalhead: Checkpoint, join_key, layer_params

export Tap, tapify, check_fixture, check_reference, reference_input, relerr

relerr(a, b) = norm(a .- b) / max(norm(b), floatmin(Float64))

const TAP_RTOL = Dict(Float16 => 2e-3, Float32 => 1e-5, Float64 => 1e-8)

"""
    reference_input(torch_shape)

Deterministic input matching `reference_input` in porting/common.py, in Julia layout.
"""
function reference_input(torch_shape)
    x = Array{Float32}(undef, reverse(torch_shape)...)
    for I in CartesianIndices(x)
        w, h, c, b = Tuple(I) .- 1
        x[I] = Float32(mod(b * 131 + c * 71 + h * 37 + w * 17, 97) / 24 - 2)
    end
    return x
end

"""
    Tap(module, name, sink)

Wraps a module of a ported model, recording each output in `sink[name]`.
"""
struct Tap{M}
    _m::M
    _name::String
    _sink::Dict{String, Vector{Any}}
end
function (t::Tap)(args...; kws...)
    y = getfield(t, :_m)(args...; kws...)
    push!(get!(getfield(t, :_sink), getfield(t, :_name), Any[]), y)
    return y
end
Base.getproperty(t::Tap, s::Symbol) = getproperty(getfield(t, :_m), s)
Flux.@layer Tap

"""
    tapify(model, names, sink) -> model

Rebuild `model` with a [`Tap`](@ref) around every module whose PyTorch name is in `names`.
Fails for structs whose fields are not type parameters (use `@concrete`).
"""
function tapify(x, names, sink, key = "")
    y = if layer_params(x) !== nothing || x isa AbstractArray || x isa Function ||
           Functors.isleaf(x)
        x
    elseif x isa Chain
        layers = tapchildren(x.layers, names, sink, key)
        layers isa Tuple ? Chain(layers...) : Chain(layers)
    elseif x isa Union{Tuple, AbstractVector, NamedTuple}
        tapchildren(x, names, sink, key)
    else
        children, re = Functors.functor(x)
        re(tapchildren(children, names, sink, key))
    end
    return key in names && y !== nothing ? Tap(y, key, sink) : y
end

function tapchildren(t::Tuple, names, sink, key)
    return ntuple(i -> tapify(t[i], names, sink, join_key(key, i - 1)), length(t))
end
function tapchildren(v::AbstractVector, names, sink, key)
    return [tapify(c, names, sink, join_key(key, i - 1)) for (i, c) in enumerate(v)]
end
function tapchildren(nt::NamedTuple, names, sink, key)
    return NamedTuple{keys(nt)}(map(k -> tapify(nt[k], names, sink, join_key(key, k)), keys(nt)))
end

function parse_tap(key)
    m = match(r"^tap/(.*)#(\d+)(?:\.(\d+))?$", key)
    return m.captures[1], parse(Int, m.captures[2]),
    m.captures[3] === nothing ? nothing : parse(Int, m.captures[3])
end

"""
    check_fixture(dir; rtol = 1e-8, model = create_model(dir; pretrained = true, eltype = Float64)) -> NamedTuple

Load the tiny fixture in `dir` with `create_model(dir; pretrained = true, eltype = Float64)` and
compare its logits and features (float64, relative error `rtol`) and every recorded module output
with PyTorch's. Module outputs are stored in float16 or float32 and compared at a tolerance set
by their precision. Returns the relative errors, with the taps in PyTorch call order.
"""
function check_fixture(dir; rtol = 1e-8, model = create_model(dir; pretrained = true, eltype = Float64))
    acts = Checkpoint(joinpath(dir, "acts.safetensors"))
    order = Metalhead.JSON.parse(SafeTensors.deserialize(joinpath(dir, "acts.safetensors")).metadata["call_order"])
    x = acts["input"]
    output = relerr(model(x), acts["output"])
    features = relerr(forward_features(model, x), acts["features"])
    tapkeys = filter(startswith("tap/"), collect(keys(acts)))
    names = Set(first(parse_tap(k)) for k in tapkeys)
    sink = Dict{String, Vector{Any}}()
    tapify(model, names, sink)(x)
    taps = map(order) do k
        name, call, part = parse_tap("tap/" * k)
        ref = acts["tap/" * k]
        got = get(sink, name, Any[])
        y = call <= length(got) ? got[call] : nothing
        y = part === nothing || y === nothing ? y : y[part]
        err = y === nothing ? missing : size(y) == size(ref) ? relerr(y, ref) : Inf
        return (; name = k, err, tol = TAP_RTOL[eltype(ref)])
    end
    return (; output,
        features,
        taps,
        pass = output <= rtol && features <= rtol &&
               all(t -> ismissing(t.err) || t.err <= t.tol, taps))
end

end # module
