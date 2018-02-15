const deps = joinpath(@__DIR__, "..", "deps")
const url = "https://github.com/FluxML/Metalhead.jl/releases/download/Models"

function getweights(name)
  mkpath(deps)
  cd(deps) do
    isfile(name) || download("$url/$name", name)
  end
end

function weights(name)
  getweights(name)
  open(deserialize, joinpath(deps, name))
end
