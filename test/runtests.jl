using Metalhead: Metalhead
using ParallelTestRunner

# Arguments select test files by path prefix (`--list` shows them all): `hub`, `models`, or one
# family as `models/resnet`; a `!` prefix excludes, e.g. `!hub`.
# METALHEAD_TEST_NETWORK=true also runs the tests that download pretrained weights.
testsuite = find_tests(@__DIR__)
foreach(helper -> delete!(testsuite, helper), ["parity", "parity_setup"])

runtests(Metalhead, ARGS; testsuite)
