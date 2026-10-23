@testsetup module ParitySetup
using Metalhead
include(joinpath(pkgdir(Metalhead), "test", "parity.jl"))
using .Parity
using JSON: JSON
using SafeTensors: SafeTensors

export Parity, check_fixture, reference_input, relerr, tapify, fixture, reference_files,
       check_param_counts

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
end
