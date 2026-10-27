using Pkg

# METALHEAD_TEST_GPU=true also runs the tests of test/gpu/ on a CUDA GPU, and adds CUDA and
# cuDNN to the test environment for them. METALHEAD_TEST_NETWORK=true also adds
# HuggingFaceTokenizers.jl, which tokenizes strings with Python's tokenizers (through
# PythonCall and CondaPkg), for the tests of tokenizers. Precompile once at the end: on
# Julia 1.13.2, `Pkg.add` fails to auto-precompile an indirect dependency whose version it
# changes (JuliaLang/Pkg.jl#4866).
const TEST_GPU = parse(Bool, get(ENV, "METALHEAD_TEST_GPU", "false"))
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
const EXTRAS = [TEST_GPU ? ["CUDA", "cuDNN"] : String[];
                TEST_NETWORK ? ["HuggingFaceTokenizers"] : String[]]
if !isempty(EXTRAS)
    withenv(() -> Pkg.add(EXTRAS), "JULIA_PKG_PRECOMPILE_AUTO" => "0")
    Pkg.precompile()
end

# Load only now: adding CUDA can change the versions of Metalhead's dependencies.
using Metalhead: Metalhead
using ParallelTestRunner

# Arguments select test files by path prefix (`--list` shows them all): `hub`, `models`, or one
# family as `models/resnet`; a `!` prefix excludes, e.g. `!hub`.
# METALHEAD_TEST_NETWORK=true also runs the tests that download pretrained weights.
testsuite = find_tests(@__DIR__)
foreach(helper -> delete!(testsuite, helper), ["parity", "parity_setup", "gpu_setup",
    "processors_setup", "data"])
TEST_GPU || filter!(((name, _),) -> !startswith(name, "gpu/"), testsuite)

runtests(Metalhead, ARGS; testsuite)
