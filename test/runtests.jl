using Metalhead: Metalhead
using CUDA
using ReTestItems

# GROUP selects what runs:
#   "All" (default)   everything
#   "Hub"             loading and API tests of the Hub ports ("Hub: ..." test items)
#   "Models"          parity tests of the ported models ("Models: ..." test items)
#   "Legacy"          the deprecated models
#   any other value   a regex over the names of the legacy test items, e.g. "AlexNet|VGG"
# Tests tagged :network download pretrained weights; enable them with METALHEAD_TEST_NETWORK=true.
const test_group = get(ENV, "GROUP", "All")
const network = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))

is_hub_port(name) = startswith(name, "Hub: ") || startswith(name, "Models: ")

function in_group(name)
    test_group == "All" && return true
    test_group == "Hub" && return startswith(name, "Hub: ")
    test_group == "Models" && return startswith(name, "Models: ")
    test_group == "Legacy" && return !is_hub_port(name)
    return !is_hub_port(name) && occursin(Regex(test_group), name)
end

runtests(ti -> in_group(ti.name) && (network || :network ∉ ti.tags), Metalhead;
    verbose_results = parse(Bool, get(ENV, "CI", "false")))
