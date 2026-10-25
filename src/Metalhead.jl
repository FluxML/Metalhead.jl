module Metalhead

using Flux
using Flux: outputsize
using Functors
using BSON
using JLD2
using Artifacts, LazyArtifacts
using Statistics
using MLUtils
using Random

import Functors

include("legacy/legacy.jl")

# Hugging Face Hub ports
using ConcreteStructs: @concrete
using Downloads: RequestError
import HuggingFaceApi as HF
using JSON: JSON
using SafeTensors: SafeTensors

include("hub/checkpoint.jl")
include("hub/download.jl")
include("hub/loadweights.jl")
include("hub/registry.jl")
include("hub/api.jl")
include("hub/images.jl")
include("hub/processors.jl")

include("Timm/Timm.jl")
include("Transformers/Transformers.jl")
include("Diffusers/Diffusers.jl")

export create_model, create_processor, list_models, forward_features, forward_head,
       forward_intermediates, feature_info

end # module
