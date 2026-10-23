@testsetup module ParitySetup
using Metalhead
include(joinpath(pkgdir(Metalhead), "test", "parity.jl"))
using .Parity
using JSON: JSON
using SafeTensors: SafeTensors
using Flux, Functors

export Parity, check_fixture, fixture_model, shrink_channels, reference_input, relerr, tapify,
       fixture, reference_files, check_param_counts, check_gradient

const TEST_DIR = joinpath(pkgdir(Metalhead), "test")

fixture(name) = joinpath(TEST_DIR, "fixtures", name)

reference_files(prefix) = filter(
    f -> startswith(basename(f), "timm--$prefix") && endswith(f, ".safetensors"),
    readdir(joinpath(TEST_DIR, "references"); join = true))

const PARAM_COUNTS = merge((JSON.parsefile(f)
for f in readdir(joinpath(TEST_DIR, "references", "param_counts"); join = true))...)

"""
    check_param_counts(arch) -> Bool

Compare the parameter count and number of `state_dict` tensors of a randomly initialised
port with those of timm (test/references/param_counts/<family>.json, from porting/param_counts.py).
"""
function check_param_counts(arch)
    model = create_model(arch)
    ref = PARAM_COUNTS[arch]
    nparams = sum(length, Metalhead.Flux.trainables(model))
    ntensors = length(first(Metalhead.torch_params(model)))
    ok = nparams == ref["num_params"] && ntensors == ref["num_tensors"]
    ok || @info "parameter counts differ from timm" arch nparams ntensors ref
    return ok
end

"""
    check_gradient(model, x) -> Bool

Whether a gradient of `sum(abs2, model(x))` in train mode reaches every trainable array of
`model` and is finite. Logs the arrays that get none (e.g. of a submodule the forward pass never
calls) or a non-finite one.
"""
function check_gradient(model, x)
    model = Flux.trainmode!(model)
    g = Flux.gradient(m -> sum(abs2, m(x)), model)[1]
    bad = String[]
    for (kp, _) in Flux.trainables(model; path = true)
        dp = g !== nothing && Functors.haskeypath(g, kp) ? Functors.getkeypath(g, kp) : nothing
        dp isa AbstractArray && all(isfinite, dp) || push!(bad, string(kp))
    end
    isempty(bad) || @info "parameters without a finite gradient" bad
    return isempty(bad)
end
end
