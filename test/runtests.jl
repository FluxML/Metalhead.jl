using Metalhead: Metalhead

# TODO account for GPU tests using name or tag filter
# TODO write GPU tests!
const test_group = get(ENV, "GROUP", "All")
const name_filter = test_group == "All" ? nothing : Regex(test_group)

@static if VERSION >= v"1.7"
    using ReTestItems
    if parse(Bool, get(ENV, "CI", "false"))
        runtests(Metalhead; name = name_filter, verbose_results = true)
    else
        # For running locally
        runtests(Metalhead; name = name_filter)
    end
else
    using TestItemRunner
    function testitem_filter(ti)
        return name_filter === nothing || match(name_filter, ti.name) !== nothing
    end
end

# Not sure why this needs to be split into a separate conditional...
@static if VERSION < v"1.7"
    @run_package_tests filter = testitem_filter
end