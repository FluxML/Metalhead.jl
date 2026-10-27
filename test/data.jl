# The test data that porting/ writes: the fixtures (test/fixtures/<name>/) and the reference
# outputs (test/references/*.safetensors). So that the package stays small,
# porting/publish.jl publishes each entry as a lazy artifact of test/Artifacts.toml, which
# is downloaded on first use. A copy in test/, not published yet, takes precedence.
module TestData
using Pkg: Pkg

export test_data, test_data_names, fixture, reference, local_data

const TEST_DIR = @__DIR__
const ARTIFACTS_TOML = joinpath(TEST_DIR, "Artifacts.toml")

"""
    test_data(path) -> String

The local path of the test data `path`, relative to test/: a fixture
(`"fixtures/resnet_basic"`) or a reference file
(`"references/timm--resnet18.a1_in1k.safetensors"`).
"""
function test_data(path::AbstractString)
    unpublished = joinpath(TEST_DIR, path)
    ispath(unpublished) && return unpublished
    Pkg.Artifacts.artifact_meta(path, ARTIFACTS_TOML) === nothing &&
        error("$path is neither in test/ nor in test/Artifacts.toml: write it with porting/ \
               and publish it with porting/publish.jl")
    # the Pkg server has only the artifacts of registered packages
    root = withenv("JULIA_PKG_SERVER" => "") do
        return Pkg.Artifacts.ensure_artifact_installed(String(path), ARTIFACTS_TOML)
    end
    return joinpath(root, basename(path))
end

fixture(name) = test_data("fixtures/$name")
reference(name) = test_data("references/$name")

"""
    local_data() -> Vector{String}

The test data in test/, to publish: written by porting/, or committed by a pull request
(test/fixtures/processors has configuration files, which stay in git).
"""
function local_data()
    fixtures = ["fixtures/$name" for name in entries("fixtures")
                if name != "processors" && isdir(joinpath(TEST_DIR, "fixtures", name))]
    references = ["$dir/$name" for dir in ("references", "references/processors")
                  for name in entries(dir) if endswith(name, ".safetensors")]
    return [fixtures; references]
end

entries(dir) = isdir(joinpath(TEST_DIR, dir)) ? readdir(joinpath(TEST_DIR, dir)) : String[]

"""
    test_data_names(dir) -> Vector{String}

The names of the entries of `dir` (`"fixtures"`, `"references"`), published or not.
"""
function test_data_names(dir::AbstractString)
    published = [basename(k)
                 for k in keys(Pkg.Artifacts.load_artifacts_toml(ARTIFACTS_TOML))
                 if dirname(k) == dir]
    return sort!(union(published, entries(dir)))
end

end # module
