using Publish
using Pkg.Artifacts

using Metalhead

# override default theme
Publish.Themes.default() = artifact"flux-theme"

p = Publish.Project(Metalhead)

# serve documentation
serve(Metalhead)