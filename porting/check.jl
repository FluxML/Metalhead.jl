# Per-module parity report of a Metalhead port against fixtures written by dump.py.
#
#   julia --project=.. check.jl specs/<name>.json ...   # the fixtures test/fixtures/<name>
#   julia --project=.. check.jl <fixture dir>           # e.g. a debug dump made with --all-taps --out
#
# A fixture that isn't in test/ is the published one (test/data.jl).
#
# Several fixtures are checked in one Julia session. A fixture that fails (or the only one given)
# gets the full table of module outputs; the others a one-line summary.
using Metalhead
include(joinpath(@__DIR__, "..", "test", "parity.jl"))
using .Parity
include(joinpath(@__DIR__, "..", "test", "data.jl"))
using .TestData: fixture

fixture_dir(arg) = isdir(arg) ? arg : fixture(splitext(basename(arg))[1])
status(err, tol) = ismissing(err) ? "not tapped" : err <= tol ? "ok" : "FAIL"

function report(arg; verbose)
    r = try
        check_fixture(fixture_dir(arg))
    catch e
        println(arg, ": ERROR")
        showerror(stdout, e)
        println("\n")
        return false
    end
    println(arg, ": ", r.pass ? "PASS" : "FAIL", "  (output ", r.output, ", features ",
        r.features, ")")
    if verbose || !r.pass
        for t in [r.taps; r.intermediates]
            println("  ", rpad(t.name, 40),
                lpad(ismissing(t.err) ? "-" : string(round(t.err; sigdigits = 2)), 10),
                "  ", status(t.err, t.tol))
        end
        println()
    end
    return r.pass
end

passed = [report(arg; verbose = length(ARGS) == 1) for arg in ARGS]
length(ARGS) > 1 && println(count(passed), " of ", length(ARGS), " fixtures pass")
exit(all(passed) ? 0 : 1)
