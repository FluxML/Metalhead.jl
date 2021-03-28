using Publish
using Pkg.Artifacts
using Metalhead

# override default theme
Publish.Themes.default() = artifact"flux-theme"

p = Publish.Project(Metalhead)

# needed to prevent error when overwriting
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)

# build documentation
deploy(Metalhead; root = "/Metalhead.jl", force = true, label = "dev")
