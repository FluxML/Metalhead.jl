using Metalhead: Metalhead
using ParallelTestRunner

# Arguments select test files by path prefix (`--list` shows them all): `hub`, `models`, one
# family as `models/resnet`, the deprecated models as `legacy`, or several, e.g.
# `legacy/efficientnet legacy/mobilenet`. A `!` prefix excludes, e.g. `!legacy`.
# METALHEAD_TEST_NETWORK=true also runs the tests that download pretrained weights;
# TEST_FAST=true tests only the smallest configurations of the deprecated models.
testsuite = find_tests(@__DIR__)
foreach(helper -> delete!(testsuite, helper), ["parity", "parity_setup", "legacy/setup"])

runtests(Metalhead, ARGS; testsuite)
