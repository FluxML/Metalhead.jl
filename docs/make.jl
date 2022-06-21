using Pkg

Pkg.develop(; path = "..")

using Publish
using Artifacts, LazyArtifacts
using Metalhead

# override default theme
cp(artifact"flux-theme", "../_flux-theme"; force = true)

p = Publish.Project(Metalhead)

function build_and_deploy(label)
    rm(label; recursive = true, force = true)
    return deploy(Metalhead; root = "/Metalhead.jl", label = label)
end
