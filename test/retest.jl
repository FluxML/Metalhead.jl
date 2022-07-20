# To run this file locally,
# use TestEnv.jl to activate the testing environment
# Then include this file

using ReTest, Metalhead
ReTest.hijack(Metalhead; include = :static)
