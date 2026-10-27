module ParitySetup
using Metalhead
include(joinpath(pkgdir(Metalhead), "test", "parity.jl"))
using .Parity
using JSON: JSON
using SafeTensors: SafeTensors
using Flux, Functors
using Test
using ChainRulesCore: unthunk

export Parity, check_fixture, check_lm_reference, check_image_reference, check_sam3_reference,
       reference_model,
       check_pipeline_fixture, check_diffusers_reference, fixture_model, fixture_inputs,
       shrink_channels,
       reference_input, relerr, tapify, fixture, reference_files, reference_files_of,
       check_param_counts,
       check_gradient, check_half_precision, JSON, SafeTensors, TEST_NETWORK, TEST_LARGE,
       check_intermediate_api, IdsTokenizer

# The reference tests download pretrained weights from the Hub.
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
# With METALHEAD_TEST_NETWORK, also those of multi-gigabyte checkpoints (FLUX.2).
const TEST_LARGE = TEST_NETWORK && parse(Bool, get(ENV, "METALHEAD_TEST_LARGE", "false"))

const TEST_DIR = joinpath(pkgdir(Metalhead), "test")

fixture(name) = joinpath(TEST_DIR, "fixtures", name)

function reference_files(prefix; org = "timm")
    return filter(
        f -> startswith(basename(f), "$org--$prefix") && endswith(f, ".safetensors"),
        readdir(joinpath(TEST_DIR, "references"); join = true))
end

# The reference files of the checkpoints registered for `arch`, which a repository prefix
# can't single out when an organisation publishes several architectures (Qwen).
function reference_files_of(arch)
    return [joinpath(TEST_DIR, "references", replace(repo, "/" => "--") * ".safetensors")
            for (repo, _) in Metalhead.lookup(arch).references]
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

# The sum of squares of an output, or of all the fields of a NamedTuple of outputs.
sum_abs2(y) = sum(abs2, y)
sum_abs2(y::NamedTuple) = sum(sum_abs2, values(y))

"""
    check_gradient(model, x...; unused = (), kwargs...) -> Bool

Whether a gradient of `sum(abs2, model(x...; kwargs...))` (summed over the fields of a
NamedTuple output) in train mode reaches every trainable array of `model` and is finite. Logs
the arrays that get none (e.g. of a submodule the forward pass never calls) or a non-finite one.
The arrays of the submodules in `unused`, which the original model doesn't use either, are
not checked.
"""
function check_gradient(model, x...; unused = (), kwargs...)
    model = Flux.trainmode!(model)
    g = Flux.gradient(m -> sum_abs2(m(x...; kwargs...)), model)[1]
    skip = IdDict(a => true for u in unused for a in Flux.trainables(u))
    bad = String[]
    for (kp, a) in Flux.trainables(model; path = true)
        haskey(skip, a) && continue
        dp = g !== nothing && Functors.haskeypath(g, kp) ? Functors.getkeypath(g, kp) :
             nothing
        # broadcast, as a gradient can wrap a GPU array lazily (NNlib.BatchedTranspose)
        dp isa AbstractArray && all(isfinite.(dp)) || push!(bad, string(kp))
    end
    isempty(bad) || @info "parameters without a finite gradient" bad
    return isempty(bad)
end

"""
    check_half_precision(model, x...; T = Float16, rtol = 1e-2, kwargs...) -> Bool

Whether the Float32 `model` converted to `T` (by `f16` or `bf16`) runs in mixed precision on
the inputs `x` (floating-point arrays are converted to `T`) and keyword arguments `kwargs`:
the layers with a conversion method of their own (the norms, such as
`Flux.f16(::LayerNorm2d, cx)`) keep `Float32` arrays and every other floating-point array is
`T`, the output has eltype `T`, no warning other than a deprecation is logged (such as
Flux's eltype mismatch), and the relative error of the output against `model(x)` is at most
`rtol`. A NamedTuple output is checked field by field. Logs what fails.
"""
function check_half_precision(model, x...; T = Float16, rtol = 1e-2, kwargs...)
    cast = Metalhead.ELTYPE_CONVERTERS[T]
    half = cast(model)
    keeps_f32(l) = which(cast, Tuple{typeof(l), Any}) !== which(cast, Tuple{Any, Any})
    wrong = String[]
    function walk(l, path)
        if keeps_f32(l)
            all(a -> !(a isa AbstractArray{<:AbstractFloat}) || eltype(a) == Float32,
                Functors.fleaves(l)) || push!(wrong, path)
        elseif l isa AbstractArray{<:AbstractFloat}
            eltype(l) == T || push!(wrong, path)
        elseif !Functors.isleaf(l)
            foreach(((k, c),) -> walk(c, "$path.$k"), pairs(Functors.children(l)))
        end
    end
    walk(half, "model")
    tohalf(a) = a isa AbstractArray{<:AbstractFloat} ? cast(a) : a
    logs, y = Test.collect_test_logs(; min_level = Base.CoreLogging.Warn) do
        return half(map(tohalf, x)...; kwargs...)
    end
    outputs(y) = y isa NamedTuple ? values(y) : (y,)
    err = maximum(((a, b),) -> relerr(Float32.(a), b),
        zip(outputs(y), outputs(model(x...; kwargs...))))
    output = eltype.(outputs(y))
    # deprecations, such as Base's of `disable_sigint` on Julia 1.14, aren't about precision
    warnings = [l.message for l in logs if l.group !== :depwarn]
    ok = isempty(wrong) && all(==(T), output) && isempty(warnings) && err <= rtol
    ok || @info "not in mixed precision" T wrong output warnings err
    return ok
end

function check_half_precision(name::AbstractString; kwargs...)
    args, inputs = fixture_inputs(Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors")))
    args = map(a -> a isa AbstractArray{<:AbstractFloat} ? Float32.(a) : a, args)
    model = fixture_model(fixture(name); eltype = Float32)
    return check_half_precision(model, args...; inputs..., kwargs...)
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

"""
    IdsTokenizer(tokens::Dict{String, Int})

A tokenizer for the processors of tiny models, through the internal interface of
Metalhead's tokenizers: a text is the token ids it writes, from 0, and the tokens of
`tokens` (`"<|image_pad|>" => 60`); the padding token is 0.
"""
struct IdsTokenizer
    tokens::Dict{String, Int}
end
function Metalhead.encode_ids(tok::IdsTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    return [get(() -> parse(Int, m.match), tok.tokens, m.match)
            for m in eachmatch(r"<\|\w+\|>|\d+", text)]
end
Metalhead.token_id(tok::IdsTokenizer, token::AbstractString) = get(tok.tokens, token, nothing)
Metalhead.pad_token_id(::IdsTokenizer) = 0
end
