using TestItems, ReTestItems
using Metalhead: Metalhead

# TODO account for GPU tests using name or tag filter
# TODO write GPU tests!
test_group = get(ENV, "GROUP", "All")
name_filter = test_group == "All" ? nothing : Regex(test_group)
ReTestItems.runtests(Metalhead; name = name_filter)