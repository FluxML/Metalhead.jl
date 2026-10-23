# Numerical parity of Metalhead ports against PyTorch, on the fixtures written by porting/dump.py
# and porting/dump_reference.py. Used by the test suite and by porting/check.jl.
module Parity

using Metalhead, Flux, Functors, LinearAlgebra, SafeTensors
using Metalhead: Checkpoint, join_key, layer_params

export Tap, tapify, check_fixture, fixture_model, shrink_channels, check_reference,
       reference_input, relerr

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
    shrink_channels(model, k; keep = ()) -> model

Rebuild the `Conv`, `BatchNorm` and `Dense` layers of `model` with every channel count
divided by `k`, like `shrunk_channels` in porting/common.py does for the PyTorch model. A count
in `keep` stays as is (the image channels and `num_classes`), every other count must be
divisible by `k`. Strides, padding, dilation, activations and `eps` are kept, and so is `groups` of a
depthwise convolution (`groups == in channels`); other `groups` stay as they are. The new
layers are zero-initialised, to be filled by `load_state!`. Fails on any other parameter.
"""
function shrink_channels(model, k::Integer; keep = ())
    function shrink(n)
        n in keep && return n
        n % k == 0 || error("channel count $n is not divisible by shrink = $k")
        return n ÷ k
    end
    isshrinkable(x) = x isa Union{Conv, BatchNorm, Dense}
    return fmap(model; exclude = x -> isshrinkable(x) || x isa AbstractArray{<:Number}) do l
        l isa Conv ? shrink_conv(l, shrink) :
        l isa BatchNorm ? shrink_batchnorm(l, shrink) :
        l isa Dense ? shrink_dense(l, shrink) :
        error("cannot shrink a parameter array of size $(size(l))")
    end
end

function shrink_conv(l::Conv, shrink)
    (; weight, bias, groups) = l
    cin, cout = size(weight, ndims(weight) - 1) * groups, size(weight, ndims(weight))
    new_cin, new_cout = shrink(cin), shrink(cout)
    new_groups = groups > 1 && groups == cin ? new_cin : groups
    weight = zeros(
        eltype(weight), size(weight)[1:(end - 2)]..., new_cin ÷ new_groups, new_cout)
    bias = bias isa AbstractArray ? zeros(eltype(bias), new_cout) : bias
    return Conv(l.σ, weight, bias, l.stride, l.pad, l.dilation, new_groups, l.pad_mode)
end

function shrink_batchnorm(l::BatchNorm, shrink)
    new = BatchNorm(shrink(l.chs), l.λ; eps = l.ϵ, momentum = l.momentum,
        affine = l.affine, track_stats = l.track_stats)
    return Metalhead.ELTYPE_CONVERTERS[eltype(l.γ)](new)
end

function shrink_dense(l::Dense, shrink)
    out, in = size(l.weight)
    new = Dense(shrink(in) => shrink(out), l.σ; bias = l.bias isa AbstractArray)
    return Metalhead.ELTYPE_CONVERTERS[eltype(l.weight)](new)
end

"""
    fixture_model(dir; eltype = Float64)

The model of the fixture in `dir` with the weights of `model.safetensors`, in test mode. For
a fixture made from a `"shrink": k` spec, the model of `config.json` is built at full size,
shrunk with [`shrink_channels`](@ref) and then loaded.
"""
function fixture_model(dir; eltype = Float64)
    config = Metalhead.JSON.parsefile(joinpath(dir, "config.json"))
    haskey(config, "shrink") || return create_model(dir; pretrained = true, eltype)
    keep = (config["pretrained_cfg"]["input_size"][1], config["num_classes"])
    model = shrink_channels(create_model(dir), config["shrink"]; keep)
    model = Metalhead.ELTYPE_CONVERTERS[eltype](model)
    spec = Metalhead.lookup(:timm, config["architecture"])
    Metalhead.load_state!(model, Checkpoint(joinpath(dir, "model.safetensors"));
        ignore = [Metalhead.DEFAULT_IGNORE; spec.ignore])
    return Flux.testmode!(model)
end

"""
    check_fixture(dir; rtol = 1e-8, model = fixture_model(dir)) -> NamedTuple

Load the tiny fixture in `dir` with [`fixture_model`](@ref) (in float64) and
compare its logits and features (float64, relative error `rtol`) and every recorded module output
with PyTorch's. Module outputs are stored in float16 or float32 and compared at a tolerance set
by their precision, and so are the `forward_intermediates` (see [`check_intermediates`](@ref)).
Returns the relative errors, with the taps in PyTorch call order.
"""
function check_fixture(dir; rtol = 1e-8, model = fixture_model(dir))
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
    intermediates = check_intermediates(model, x, acts)
    return (; output,
        features,
        taps,
        intermediates,
        pass = output <= rtol && features <= rtol &&
               all(t -> ismissing(t.err) || t.err <= t.tol, taps) &&
               all(t -> t.err <= t.tol, intermediates))
end

"""
    check_intermediates(model, x, acts) -> Vector{NamedTuple}

Compare `forward_intermediates(model, x)` (plain, with `norm = true` and the prefix tokens of
vision transformers) with the references "intermediates/<i>", "intermediates_norm/<i>" and
"intermediates_prefix/<i>" of a fixture, and its features with "features". Without
"intermediates_norm", the normed intermediates must equal the plain references.
"""
function check_intermediates(model, x, acts)
    n = count(startswith("intermediates/"), keys(acts))
    n == 0 && return NamedTuple[]
    errs = NamedTuple[]
    record!(name, y, ref, tol) = push!(errs,
        (; name, err = size(y) == size(ref) ? relerr(y, ref) : Inf, tol))
    tol = TAP_RTOL[Float32]
    features, plain = forward_intermediates(model, x)
    record!("features", features, acts["features"], 1e-8)
    _, normed = forward_intermediates(model, x; norm = true)
    length(plain) == length(normed) == n || return [(; name = "count", err = Inf, tol)]
    for i in 1:n
        record!("intermediates/$i", plain[i], acts["intermediates/$i"], tol)
        key = haskey(acts, "intermediates_norm/$i") ? "intermediates_norm/$i" :
              "intermediates/$i"
        record!("intermediates_norm/$i", normed[i], acts[key], tol)
    end
    if haskey(acts, "intermediates_prefix/1")
        _, pairs = forward_intermediates(model, x; return_prefix_tokens = true)
        for i in 1:n
            record!("intermediates_prefix/$i", last(pairs[i]),
                acts["intermediates_prefix/$i"], tol)
        end
    end
    return errs
end

end # module
