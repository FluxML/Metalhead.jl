using Pkg

Pkg.develop(; path = "..")

using Revise
using Publish
using Artifacts, LazyArtifacts

using Metalhead

# override default theme
cp(artifact"flux-theme", "../_flux-theme"; force = true)

p = Publish.Project(Metalhead)

# serve documentation
serve(Metalhead)
