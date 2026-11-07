"""
    Metalhead

Flux ports of the models of the Hugging Face Hub: the vision models of timm, the language,
text, vision-language and segmentation models of transformers, the text-to-image pipelines
of diffusers and the text embeddings of sentence-transformers. [`create_model`](@ref) builds
a model, with its weights, and [`create_processor`](@ref) its pre- and post-processing.
"""
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
using ChainRulesCore: ChainRulesCore
using Downloads: RequestError
import HuggingFaceApi as HF
using JSON: JSON
using SafeTensors: SafeTensors

include("hub/checkpoint.jl")
include("hub/download.jl")
include("hub/loadweights.jl")
include("hub/initialization.jl")
include("hub/registry.jl")
include("hub/api.jl")
include("hub/images.jl")
include("hub/processors.jl")
include("hub/tokenizers.jl")

include("Timm/Timm.jl")
include("Transformers/Transformers.jl")
include("Diffusers/Diffusers.jl")
include("SentenceTransformers/SentenceTransformers.jl")

using .SentenceTransformers: encode_query, encode_document, similarity
using .Transformers: generate

export create_model, create_processor, list_models, forward_features, forward_head,
       forward_intermediates, feature_info, post_process_object_detection,
       post_process_instance_segmentation, post_process_semantic_segmentation,
       encode_query, encode_document, similarity, generate

# `public` is a syntax error before Julia 1.11.
VERSION >= v"1.11.0-DEV.469" && eval(Expr(:public,
    :Timm, :Transformers, :Diffusers, :SentenceTransformers,
    :TextProcessor, :image_array, :image_matrix,
    # the porting API, docs/src/api/porting.md
    :ArchSpec, :register!, :AbstractConfigFormat, :architecture_of, :model_args,
    :skip_prefix, :weights_name, :instantiate, :Checkpoint, :load_state!))

end # module
