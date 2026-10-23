module ParitySetup
using Metalhead
include(joinpath(pkgdir(Metalhead), "test", "parity.jl"))
using .Parity
using JSON: JSON
using SafeTensors: SafeTensors
using Flux, Functors
using Test
using ChainRulesCore: unthunk

export Parity, check_fixture, fixture_model, shrink_channels, reference_input, relerr,
       tapify,
       fixture, reference_files, check_param_counts, check_gradient, JSON, SafeTensors,
       TEST_NETWORK, check_intermediate_api

# The reference tests download pretrained weights from the Hub.
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))

const TEST_DIR = joinpath(pkgdir(Metalhead), "test")

fixture(name) = joinpath(TEST_DIR, "fixtures", name)

function reference_files(prefix)
    return filter(
        f -> startswith(basename(f), "timm--$prefix") && endswith(f, ".safetensors"),
        readdir(joinpath(TEST_DIR, "references"); join = true))
end

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
        dp = g !== nothing && Functors.haskeypath(g, kp) ? Functors.getkeypath(g, kp) :
             nothing
        dp isa AbstractArray && all(isfinite, dp) || push!(bad, string(kp))
    end
    isempty(bad) || @info "parameters without a finite gradient" bad
    return isempty(bad)
end

function check_intermediate_api(name::AbstractString)
    model = fixture_model(fixture(name); eltype = Float32)
    x = Float32.(Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))["input"])
    return check_intermediate_api(model, x)
end

function check_intermediate_api(model, x)
    @testset "forward_intermediates" begin
        info = feature_info(model)
        n = length(info)
        features, ys = forward_intermediates(model, x)
        @test features ≈ forward_features(model, x)
        @test length(ys) == n
        @test size.(ys, 3) == [f.num_chs for f in info]
        @test forward_intermediates(model, x; norm = true)[2] ≈ ys
        @test forward_intermediates(model, x; indices = 1)[2] ≈ ys[n:n]
        @test forward_intermediates(model, x; indices = [n, 1, 1],
            intermediates_only = true) ≈ ys[[1, n]]
        early, selected = forward_intermediates(model, x; indices = (1,), stop_early = true)
        @test early ≈ only(selected) ≈ ys[1]
        complete, selected = forward_intermediates(model, x; indices = [n], stop_early = true)
        @test complete ≈ features
        @test only(selected) ≈ ys[n]
        for indices in (0, n + 1, -1, [], [0], [n + 1])
            @test_throws "indices" forward_intermediates(model, x; indices)
        end
        @test_throws "output_fmt" forward_intermediates(model, x; output_fmt = "NLC")
        g = only(Flux.gradient(model) do m
            sum(abs2,
                only(forward_intermediates(m, x; indices = [2], stop_early = true,
                    intermediates_only = true)))
        end)
        leaves = map(Flux.trainables(model; path = true)) do (kp, _)
            dp = Functors.haskeypath(g, kp) ? Functors.getkeypath(g, kp) : nothing
            return unthunk(dp)
        end
        grads = filter(a -> a isa AbstractArray, leaves)
        @test 0 < length(grads) < length(Flux.trainables(model))
        @test all(a -> all(isfinite, a), grads)
        @test any(a -> any(!iszero, a), grads)
    end
end
end
