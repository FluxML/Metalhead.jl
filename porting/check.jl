# Per-module parity report of a Metalhead port against a fixture written by dump.py.
#
#   julia --project=.. check.jl specs/<name>.json   # the committed fixture test/fixtures/<name>
#   julia --project=.. check.jl <fixture dir>       # e.g. a debug dump made with --all-taps --out
using Metalhead
include(joinpath(@__DIR__, "..", "test", "parity.jl"))
using .Parity

arg = only(ARGS)
dir = isdir(arg) ? arg :
      joinpath(@__DIR__, "..", "test", "fixtures", splitext(basename(arg))[1])
r = check_fixture(dir)

status(err, tol) = ismissing(err) ? "not tapped" : err <= tol ? "ok" : "FAIL"
println("output    relerr = ", r.output)
println("features  relerr = ", r.features)
println()
for t in r.taps
    println(rpad(t.name, 40),
        lpad(ismissing(t.err) ? "-" : string(round(t.err; sigdigits = 2)), 10),
        "  ", status(t.err, t.tol))
end
println()
println(r.pass ? "PASS" : "FAIL")
exit(r.pass ? 0 : 1)
