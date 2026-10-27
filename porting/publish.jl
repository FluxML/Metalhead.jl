# Publish the test data in test/: each fixture directory (test/fixtures/<name>/) and
# reference file (test/references/**/*.safetensors), written by porting/ or committed by a
# pull request, becomes a lazy artifact, a tarball on the Hub dataset
# FluxML/metalhead-test-data bound in test/Artifacts.toml. The published copies are then
# removed from test/: the tests download them on first use (test/data.jl), and the
# artifacts are already in the depot.
#
#   julia publish.jl               # every entry in test/, uploaded in one commit
#   julia publish.jl --no-upload   # bind only, and upload out/test-data/ later with `hf upload`
#
# Uploading needs write access to the dataset (`hf auth login`, or HF_TOKEN), and uv, which
# runs huggingface_hub. Without it, commit the data instead: once the pull request is
# merged, .github/workflows/publish-test-data.yml runs this script. Tarballs are named by
# their sha256, which test/Artifacts.toml pins, so none is overwritten.
using Pkg, SHA
using Pkg.Artifacts: create_artifact, archive_artifact, bind_artifact!, artifact_meta

const REPO = "FluxML/metalhead-test-data"
const TEST_DIR = normpath(joinpath(@__DIR__, "..", "test"))
const ARTIFACTS_TOML = joinpath(TEST_DIR, "Artifacts.toml")
const STAGING = joinpath(@__DIR__, "out", "test-data")
include(joinpath(TEST_DIR, "data.jl"))

# `path` as an artifact in the depot, and its tarball in STAGING unless it's already bound
function pack(path)
    src = joinpath(TEST_DIR, path)
    hash = create_artifact(dir -> cp(src, joinpath(dir, basename(path))))
    meta = artifact_meta(path, ARTIFACTS_TOML)
    meta !== nothing && meta["git-tree-sha1"] == bytes2hex(hash.bytes) && return nothing
    tmp = tempname() * ".tar.gz"
    archive_artifact(hash, tmp)
    sha = bytes2hex(open(sha256, tmp))
    file = "$(chopsuffix(path, ".safetensors"))/$sha.tar.gz"
    mkpath(dirname(joinpath(STAGING, file)))
    mv(tmp, joinpath(STAGING, file); force = true)
    url = "https://huggingface.co/datasets/$REPO/resolve/main/$file"
    return (; path, hash, url, sha)
end

const UPLOAD = """
import sys
from huggingface_hub import upload_folder
repo, folder, message = sys.argv[1:]
print(upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder,
                    commit_message=message))
"""

function main(args)
    paths = TestData.local_data()
    isempty(paths) && return println("nothing to publish in test/")
    packed = filter(!isnothing, map(pack, paths))
    if "--no-upload" ∉ args && !isempty(packed)
        message = "Publish $(length(packed)) entries of test data"
        run(`uv run --no-project --with huggingface_hub python -I -c $UPLOAD $REPO $STAGING
             $message`)
        rm(STAGING; recursive = true)
    end
    for p in packed
        bind_artifact!(ARTIFACTS_TOML, p.path, p.hash; download_info = [(p.url, p.sha)],
            lazy = true, force = true)
    end
    foreach(path -> rm(joinpath(TEST_DIR, path); recursive = true), paths)
    println(length(packed), " published, ", length(paths) - length(packed), " unchanged")
    "--no-upload" in args && !isempty(packed) &&
        println("upload them with: uv run hf upload $REPO out/test-data . --repo-type dataset")
end

main(ARGS)
