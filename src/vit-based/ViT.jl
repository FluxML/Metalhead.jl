module ViT

using Flux
using Flux: outputsize, Zygote
using Functors
using BSON
using Artifacts, LazyArtifacts
using TensorCast
using Statistics

include("../utilities.jl")

include("mlpmixer.jl")

export  MLPMixer

# use Flux._big_show to pretty print large models
for T in (:MLPMixer,)
    @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end