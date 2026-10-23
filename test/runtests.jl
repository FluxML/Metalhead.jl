using Metalhead: Metalhead
using CUDA
using ReTestItems

# TODO account for GPU tests using name or tag filter
# TODO write GPU tests!
const test_group = get(ENV, "GROUP", "All")
const name_filter = test_group == "All" ? nothing : Regex(test_group)

runtests(Metalhead; name = name_filter,
         verbose_results = parse(Bool, get(ENV, "CI", "false")))
